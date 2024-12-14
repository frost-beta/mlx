// Copyright © 2023-2024 Apple Inc.

#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <sstream>

#include <dlfcn.h>
#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/compiled_preamble.h"
#include "mlx/device.h"
#include "mlx/graph_utils.h"

namespace mlx::core {

#ifdef _MSC_VER

namespace {

// Split string into array.
std::vector<std::string> str_split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// Run a command and get its output.
std::string exec(std::string cmd) {
  std::unique_ptr<FILE, decltype(&_pclose)> pipe(
      _popen(cmd.c_str(), "r"), _pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed.");
  }
  char buffer[128];
  std::string ret;
  while (fgets(buffer, sizeof(buffer), pipe.get())) {
    ret += buffer;
  }
  // Trim trailing spaces.
  ret.erase(
      std::find_if(
          ret.rbegin(),
          ret.rend(),
          [](unsigned char ch) { return !std::isspace(ch); })
          .base(),
      ret.end());
  return ret;
}

// Get path information about MSVC.
struct VisualStudioInfo {
  VisualStudioInfo() {
#ifdef _M_ARM64
    arch = "arm64";
#else
    arch = "x64";
#endif
    // Get path of Visual Studio.
    std::string vs_path = exec(fmt::format(
        "\"{0}\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
        " -property installationPath",
        std::getenv("ProgramFiles(x86)")));
    if (vs_path.empty()) {
      throw std::runtime_error("Can not find Visual Studio.");
    }
    // Read the envs from vcvarsall.
    std::string envs = exec(fmt::format(
        "\"{0}\\VC\\Auxiliary\\Build\\vcvarsall.bat\" {1} >NUL && set",
        vs_path,
        arch));
    for (const std::string& line : str_split(envs, '\n')) {
      // Each line is in the format "ENV_NAME=values".
      auto pos = line.find_first_of('=');
      if (pos == std::string::npos || pos == 0 || pos == line.size() - 1)
        continue;
      std::string name = line.substr(0, pos);
      std::string value = line.substr(pos + 1);
      if (name == "LIB") {
        libpaths = str_split(value, ';');
      } else if (name == "VCToolsInstallDir") {
        cl_exe = fmt::format("{0}\\bin\\Host{1}\\{1}\\cl.exe", value, arch);
      }
    }
  }
  std::string arch;
  std::string cl_exe;
  std::vector<std::string> libpaths;
};

const VisualStudioInfo& GetVisualStudioInfo() {
  static VisualStudioInfo info;
  return info;
}

} // namespace

#endif // _MSC_VER

struct CompilerCache {
  struct DLib {
    DLib(const std::string& libname) {
      lib = dlopen(libname.c_str(), RTLD_NOW);
      if (!lib) {
        std::ostringstream msg;
        msg << "Could not load C++ shared library " << dlerror();
        throw std::runtime_error(msg.str());
      }
    }

    ~DLib() {
      dlclose(lib);
    }
    void* lib;
  };
  // Statics to cache compiled libraries and functions
  std::list<DLib> libs;
  std::unordered_map<std::string, void*> kernels;
  std::shared_mutex mtx;
};

static CompilerCache cache{};

// GPU compile is always available if the GPU is available and since we are in
// this file CPU compile is also available.
namespace detail {
bool compile_available_for_device(const Device& device) {
  return true;
}

} // namespace detail

std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name).string();
}

// Return a pointer to a compiled function
void* compile(
    const std::string& kernel_name,
    const std::function<std::string(void)>& source_builder) {
  {
    std::shared_lock lock(cache.mtx);
    if (auto it = cache.kernels.find(kernel_name); it != cache.kernels.end()) {
      return it->second;
    }
  }

  std::unique_lock lock(cache.mtx);
  if (auto it = cache.kernels.find(kernel_name); it != cache.kernels.end()) {
    return it->second;
  }
  std::string source_code = source_builder();
  std::string kernel_file_name;

  // Deal with long kernel names. Maximum length for filename on macOS is 255
  // characters, and on Windows the maximum length for whole path is 260. Clip
  // file name with a little extra room and append a 16 character hash.
#ifdef _WIN32
  constexpr int max_file_name_length = 140;
#else
  constexpr int max_file_name_length = 245;
#endif
  if (kernel_name.size() > max_file_name_length) {
    std::ostringstream file_name;
    file_name
        << std::string_view(kernel_name).substr(0, max_file_name_length - 16);
    auto file_id =
        std::hash<std::string>{}(kernel_name.substr(max_file_name_length - 16));
    file_name << "_" << std::hex << std::setw(16) << file_id << std::dec;
    kernel_file_name = file_name.str();
  } else {
    kernel_file_name = kernel_name;
  }

  std::ostringstream shared_lib_name;
  shared_lib_name << "lib" << kernel_file_name << ".so";
  auto shared_lib_path = get_temp_file(shared_lib_name.str());
  bool lib_exists = false;
  {
    std::ifstream f(shared_lib_path.c_str());
    lib_exists = f.good();
  }

  if (!lib_exists) {
    // Open source file and write source code to it
    std::ostringstream source_file_name;
    source_file_name << kernel_file_name << ".cpp";
    auto source_file_path = get_temp_file(source_file_name.str());

    std::ofstream source_file(source_file_path);
    source_file << source_code;
    source_file.close();

#ifdef _MSC_VER
    const VisualStudioInfo& info = GetVisualStudioInfo();
    std::string libpaths;
    for (const std::string& lib : info.libpaths) {
      libpaths += fmt::format(" /libpath:\"{0}\"", lib);
    }
    std::string build_command = fmt::format(
        "\""
        "\"{0}\" /LD /EHsc /nologo /std:c++17 \"{1}\" /link /out:\"{2}\"{3}"
        "\"",
        info.cl_exe,
        source_file_path,
        shared_lib_path,
        libpaths);
#else
    std::string build_command = fmt::format(
        "g++ -std=c++17 -O3 -Wall -fPIC -shared '{0}' -o '{1}'",
        source_file_path,
        shared_lib_path);
#endif
    auto return_code = system(build_command.c_str());
    if (return_code) {
      std::ostringstream msg;
      msg << "[Compile::eval_cpu] Failed to compile function " << kernel_name
          << " with error code " << return_code << "." << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

  // load library
  cache.libs.emplace_back(shared_lib_path);

  // Load function
  void* fun = dlsym(cache.libs.back().lib, kernel_name.c_str());
  if (!fun) {
    std::ostringstream msg;
    msg << "[Compile::eval_cpu] Failed to load compiled function "
        << kernel_name << std::endl
        << dlerror();
    throw std::runtime_error(msg.str());
  }
  cache.kernels.insert({kernel_name, fun});
  return fun;
}

inline void build_kernel(
    std::ostream& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::unordered_set<uintptr_t>& constant_ids,
    bool contiguous,
    int ndim) {
  // All outputs should have the exact same shape and will be row contiguous
  auto output_shape = outputs[0].shape();
  auto output_strides = outputs[0].strides();

  // Constants are scalars that are captured by value and cannot change
  auto is_constant = [&constant_ids](const array& x) {
    return constant_ids.find(x.id()) != constant_ids.end();
  };

  NodeNamer namer;

#ifdef _MSC_VER
  // Export the symbol
  os << "__declspec(dllexport) ";
#endif

  // Start the kernel
  os << "void " << kernel_name << "(void** args) {" << std::endl;

  // Add the input arguments
  int cnt = 0;
  for (auto& x : inputs) {
    auto& xname = namer.get_name(x);

    // Skip constants from the input list
    if (is_constant(x)) {
      continue;
    }

    auto tstr = get_type_string(x.dtype());
    os << "  " << tstr << "* " << xname << " = (" << tstr << "*)args[" << cnt++
       << "];" << std::endl;
    // Scalars and contiguous need no strides
    if (!is_scalar(x) && !contiguous) {
      os << "  const size_t* " << xname << "_strides = (size_t*)args[" << cnt++
         << "];" << std::endl;
    }
  }

  // Add the output arguments
  for (auto& x : outputs) {
    auto tstr = get_type_string(x.dtype());
    os << "  " << tstr << "* " << namer.get_name(x) << " = (" << tstr
       << "*)args[" << cnt++ << "];" << std::endl;
  }
  // Add output strides and shape to extract the indices.
  if (!contiguous) {
    os << "  const int* shape = (int*)args[" << cnt++ << "];" << std::endl;
  } else {
    os << "  const size_t size = (size_t)args[" << cnt++ << "];" << std::endl;
  }

  if (contiguous) {
    os << "  for (size_t i = 0; i < size; ++i) {" << std::endl;
  } else {
    for (int d = 0; d < ndim; ++d) {
      os << "  for (int i" << d << " = 0; i" << d << " < shape[" << d
         << "]; ++i" << d << ") {" << std::endl;
    }
  }

  // Read the inputs in tmps
  for (auto& x : inputs) {
    auto& xname = namer.get_name(x);

    if (is_constant(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = ";
      print_constant(os, x);
      os << ";" << std::endl;
    } else if (is_scalar(x)) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[0];" << std::endl;
    } else if (contiguous) {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = "
         << xname << "[i];" << std::endl;
    } else {
      os << "  " << get_type_string(x.dtype()) << " tmp_" << xname << " = *"
         << xname << ";" << std::endl;
    }
  }

  // Actually write the computation
  for (auto& x : tape) {
    os << "  " << get_type_string(x.dtype()) << " tmp_" << namer.get_name(x)
       << " = ";
    if (is_static_cast(x.primitive())) {
      os << "static_cast<" << get_type_string(x.dtype()) << ">(tmp_"
         << namer.get_name(x.inputs()[0]) << ");" << std::endl;
    } else {
      x.primitive().print(os);
      os << "()(";
      for (int i = 0; i < x.inputs().size() - 1; i++) {
        os << "tmp_" << namer.get_name(x.inputs()[i]) << ", ";
      }
      os << "tmp_" << namer.get_name(x.inputs().back()) << ");" << std::endl;
    }
  }

  // Write the outputs from tmps
  for (auto& x : outputs) {
    if (contiguous) {
      os << "  " << namer.get_name(x) << "[i] = tmp_" << namer.get_name(x)
         << ";" << std::endl;
    } else {
      os << "  *" << namer.get_name(x) << "++ = tmp_" << namer.get_name(x)
         << ";" << std::endl;
    }
  }

  // Close loops
  if (contiguous) {
    os << "  }" << std::endl;
  } else {
    for (int d = ndim - 1; d >= 0; --d) {
      // Update pointers
      for (auto& x : inputs) {
        if (is_constant(x) || is_scalar(x)) {
          continue;
        }
        auto& xname = namer.get_name(x);
        os << "  " << xname << " += " << xname << "_strides[" << d << "];"
           << std::endl;
        if (d < ndim - 1) {
          os << "  " << xname << " -= " << xname << "_strides[" << d + 1 << "]"
             << " * shape[" << d + 1 << "];" << std::endl;
        }
      }
      os << "  }" << std::endl;
    }
  }

  // Finish the kernel
  os << "}" << std::endl;
}

void Compiled::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (kernel_lib_.empty()) {
    kernel_lib_ = build_lib_name(inputs_, outputs_, tape_, constant_ids_);
  }

  // Figure out which kernel we are using
  auto& shape = outputs[0].shape();
  auto contiguous = compiled_check_contiguity(inputs, shape);

  // Handle all broadcasting and collect function input arguments
  std::vector<void*> args;
  std::vector<std::vector<size_t>> strides;
  for (int i = 0; i < inputs.size(); i++) {
    // Skip constants.
    if (constant_ids_.find(inputs_[i].id()) != constant_ids_.end()) {
      continue;
    }
    auto& x = inputs[i];
    args.push_back((void*)x.data<void>());

    if (contiguous || is_scalar(x)) {
      continue;
    }

    // Broadcast the input to the output shape.
    std::vector<size_t> xstrides;
    int j = 0;
    for (; j < shape.size() - x.ndim(); j++) {
      if (shape[j] == 1) {
        xstrides.push_back(outputs[0].strides()[j]);
      } else {
        xstrides.push_back(0);
      }
    }
    for (int i = 0; i < x.ndim(); i++, j++) {
      if (x.shape(i) == 1) {
        if (shape[j] == 1) {
          xstrides.push_back(outputs[0].strides()[j]);
        } else {
          xstrides.push_back(0);
        }
      } else {
        xstrides.push_back(x.strides()[i]);
      }
    }
    strides.push_back(std::move(xstrides));
    args.push_back(strides.back().data());
  }

  // Get the kernel name from the lib
  int ndim = shape.size();
  auto kernel_name = kernel_lib_ + (contiguous ? "_contiguous" : "_strided_");
  if (!contiguous) {
    kernel_name += std::to_string(shape.size());
  }

  // Get the function
  auto fn_ptr = compile(kernel_name, [&]() {
    std::ostringstream kernel;
    kernel << get_kernel_preamble() << std::endl;
    kernel << "extern \"C\"  {" << std::endl;
    build_kernel(
        kernel,
        kernel_name,
        inputs_,
        outputs_,
        tape_,
        constant_ids_,
        contiguous,
        ndim);
    // Close extern "C"
    kernel << "}" << std::endl;
    return kernel.str();
  });

  compiled_allocate_outputs(
      inputs, outputs, inputs_, constant_ids_, contiguous, false);

  for (auto& x : outputs) {
    args.push_back(x.data<void>());
  }
  if (!contiguous) {
    args.push_back((void*)outputs[0].shape().data());
  } else {
    args.push_back((void*)outputs[0].data_size());
  }
  auto fun = (void (*)(void**))fn_ptr;
  fun(args.data());
}

} // namespace mlx::core
