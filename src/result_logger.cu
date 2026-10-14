#include "result_logger.h"
#include <cstdio>
#include <fstream>
#include <cstdlib>

namespace cmb {

ResultLogger g_results;

void ResultLogger::writeJson(const std::string &path) const {
  JsonWriter jw;
  jw.beginObject();
  jw.keyValue("device", device_name_);
  jw.keyValue("architecture", arch_);
  jw.keyValue("sm_count", sm_count_);
  jw.keyValue("clock_khz", clock_khz_);
  jw.keyValue("memory_clock_khz", mem_clock_khz_);
  jw.keyValue("memory_bus_width_bits", mem_bus_width_);

  jw.key("results");
  jw.beginObject();
  for (const auto &sec : sections_) {
    jw.key(sec.first);
    jw.beginObject();
    for (const auto &kv : sec.second) {
      jw.key(kv.first);
      jw.beginObject();
      jw.key("value");
      if (kv.second.unit == "N/A") jw.null();
      else jw.value(kv.second.value);
      jw.keyValue("unit", kv.second.unit);
      jw.endObject();
    }
    jw.endObject();
  }
  jw.endObject();
  jw.endObject();

  std::ofstream ofs(path);
  if (!ofs) { fprintf(stderr, "Cannot open results path: %s\n", path.c_str()); std::exit(EXIT_FAILURE); }
  ofs << jw.str();
  if (!ofs) { fprintf(stderr, "Cannot write results path: %s\n", path.c_str()); std::exit(EXIT_FAILURE); }
}

} // namespace cmb
