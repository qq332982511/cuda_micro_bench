#pragma once

#include <map>
#include <string>
#include <vector>

namespace cmb {

// Minimal JSON writer for benchmark results.
class JsonWriter {
public:
  JsonWriter() = default;

  void beginObject() {
    sep();
    out_ += "{\n";
    indent_ += 2;
    need_comma_ = false;
  }
  void endObject() {
    indent_ -= 2;
    trimComma();
    out_ += std::string(indent_, ' ') + "}\n";
    need_comma_ = true;
  }

  void beginArray() {
    sep();
    out_ += "[\n";
    indent_ += 2;
    need_comma_ = false;
  }
  void endArray() {
    indent_ -= 2;
    trimComma();
    out_ += std::string(indent_, ' ') + "]\n";
    need_comma_ = true;
  }

  void key(const std::string &k) {
    sep();
    out_ += std::string(indent_, ' ') + "\"" + k + "\": ";
  }

  void value(const std::string &v) { out_ += "\"" + escape(v) + "\"\n"; need_comma_ = true; }
  void value(double v) { out_ += std::to_string(v) + "\n"; need_comma_ = true; }
  void value(int v) { out_ += std::to_string(v) + "\n"; need_comma_ = true; }
  void value(bool v) { out_ += std::string(v ? "true" : "false") + "\n"; need_comma_ = true; }
  void null() { out_ += "null\n"; need_comma_ = true; }

  void keyValue(const std::string &k, const std::string &v) {
    key(k);
    value(v);
  }
  void keyValue(const std::string &k, double v) {
    key(k);
    value(v);
  }
  void keyValue(const std::string &k, int v) {
    key(k);
    value(v);
  }
  void keyValue(const std::string &k, bool v) {
    key(k);
    value(v);
  }

  std::string str() const { return out_; }

private:
  std::string out_;
  int indent_ = 0;
  bool need_comma_ = false;

  void sep() {
    if (need_comma_) {
      // Replace the trailing newline after the previous element with a comma
      // and start a new indented line for the next element.
      if (!out_.empty() && out_.back() == '\n')
        out_.pop_back();
      out_ += ",\n";
      need_comma_ = false;
    }
  }

  void trimComma() {
    size_t pos = out_.find_last_not_of(" \n\t");
    if (pos != std::string::npos && out_[pos] == ',') {
      out_.erase(pos);
      out_ += "\n";
    }
  }

  std::string escape(const std::string &s) const {
    std::string r;
    for (char c : s) {
      switch (c) {
      case '"': r += "\\\""; break;
      case '\\': r += "\\\\"; break;
      case '\n': r += "\\n"; break;
      case '\r': r += "\\r"; break;
      case '\t': r += "\\t"; break;
      default: r += c;
      }
    }
    return r;
  }
};

// Simple key/value result logger.
class ResultLogger {
public:
  void setDevice(const std::string &name, const std::string &arch, int sm_count,
                 int clock_khz, int mem_clock_khz, int mem_bus_width) {
    device_name_ = name;
    arch_ = arch;
    sm_count_ = sm_count;
    clock_khz_ = clock_khz;
    mem_clock_khz_ = mem_clock_khz;
    mem_bus_width_ = mem_bus_width;
  }

  void log(const std::string &section, const std::string &key,
           double value, const std::string &unit = "") {
    sections_[section][key] = {value, unit};
  }

  void writeJson(const std::string &path) const;
  bool hasDevice() const { return !device_name_.empty(); }

private:
  struct Entry {
    double value;
    std::string unit;
  };
  std::string device_name_, arch_;
  int sm_count_ = 0;
  int clock_khz_ = 0;
  int mem_clock_khz_ = 0;
  int mem_bus_width_ = 0;
  std::map<std::string, std::map<std::string, Entry>> sections_;
};

extern ResultLogger g_results;

} // namespace cmb
