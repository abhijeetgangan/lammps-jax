// Loads bundle JSON from export.py; first-quoted-key lookup, safe under the fixed sorted schema.

#include "lammps_jax_model.h"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace lammps_jax {
namespace {

size_t find_key(const std::string &json, const std::string &key)
{
  const std::string quoted = "\"" + key + "\"";
  const size_t key_pos = json.find(quoted);
  if (key_pos == std::string::npos) throw std::runtime_error("Missing key in bundle: " + key);
  const size_t colon = json.find(':', key_pos + quoted.size());
  if (colon == std::string::npos) throw std::runtime_error("Malformed key in bundle: " + key);
  return colon + 1;
}

void skip_ws(const std::string &json, size_t &pos)
{
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
}

std::string get_string(const std::string &json, const std::string &key)
{
  size_t pos = find_key(json, key);
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '"') throw std::runtime_error("Expected JSON string");
  ++pos;
  std::string out;
  while (pos < json.size()) {
    const char c = json[pos++];
    if (c == '"') return out;
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (pos >= json.size()) throw std::runtime_error("Invalid JSON escape");
    const char esc = json[pos++];
    switch (esc) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      default:
        throw std::runtime_error("Unsupported JSON escape in model bundle");
    }
  }
  throw std::runtime_error("Unterminated JSON string");
}

double get_number(const std::string &json, const std::string &key)
{
  size_t pos = find_key(json, key);
  skip_ws(json, pos);
  const size_t end = json.find_first_of(",}\n\r\t ", pos);
  return std::stod(json.substr(pos, end - pos));
}

// Bundles written before the key existed keep the fallback value.
int get_int_or(const std::string &json, const std::string &key, int fallback)
{
  const std::string quoted = "\"" + key + "\"";
  if (json.find(quoted) == std::string::npos) return fallback;
  return static_cast<int>(get_number(json, key));
}

// Flat string-array parse; no escapes since values are identifier-like target names.
std::vector<std::string> get_string_array_or(const std::string &json, const std::string &key)
{
  const std::string quoted = "\"" + key + "\"";
  if (json.find(quoted) == std::string::npos) return {};
  size_t pos = find_key(json, key);
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '[')
    throw std::runtime_error("Expected JSON array for key: " + key);
  ++pos;
  std::vector<std::string> values;
  for (;;) {
    skip_ws(json, pos);
    if (pos >= json.size()) throw std::runtime_error("Unterminated JSON array for key: " + key);
    if (json[pos] == ']') break;
    if (json[pos] != '"')
      throw std::runtime_error("Expected string in JSON array for key: " + key);
    const size_t end = json.find('"', pos + 1);
    if (end == std::string::npos)
      throw std::runtime_error("Unterminated string in JSON array for key: " + key);
    values.push_back(json.substr(pos + 1, end - pos - 1));
    pos = end + 1;
    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == ',') ++pos;
  }
  return values;
}

// Parses a flat JSON array of integers; missing keys yield an empty vector.
std::vector<int> get_int_array_or(const std::string &json, const std::string &key)
{
  const std::string quoted = "\"" + key + "\"";
  if (json.find(quoted) == std::string::npos) return {};
  size_t pos = find_key(json, key);
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '[')
    throw std::runtime_error("Expected JSON array for key: " + key);
  ++pos;
  std::vector<int> values;
  for (;;) {
    skip_ws(json, pos);
    if (pos >= json.size()) throw std::runtime_error("Unterminated JSON array for key: " + key);
    if (json[pos] == ']') break;
    const size_t end = json.find_first_of(",]", pos);
    if (end == std::string::npos)
      throw std::runtime_error("Unterminated JSON array for key: " + key);
    values.push_back(std::stoi(json.substr(pos, end - pos)));
    pos = end;
    if (json[pos] == ',') ++pos;
  }
  return values;
}

bool get_bool(const std::string &json, const std::string &key)
{
  size_t pos = find_key(json, key);
  skip_ws(json, pos);
  if (json.compare(pos, 4, "true") == 0) return true;
  if (json.compare(pos, 5, "false") == 0) return false;
  throw std::runtime_error("Expected boolean for key: " + key);
}

// Validation only: the sole supported layout needs no runtime representation.
void parse_input_layout(const std::string &value)
{
  if (value == "sparse-edge") return;
  throw std::runtime_error("Unsupported LAMMPS-JAX input layout '" + value +
                           "'; re-export the bundle with the current lammps_jax exporter");
}

ForceLayout parse_force_layout(const std::string &value)
{
  if (value == "atom-force") return ForceLayout::Atom;
  if (value == "edge-force") return ForceLayout::Edge;
  throw std::runtime_error("Unsupported LAMMPS-JAX force output layout '" + value + "'");
}

NewtonMode parse_newton_mode(const std::string &value)
{
  if (value == "any") return NewtonMode::Any;
  if (value == "on") return NewtonMode::On;
  if (value == "off") return NewtonMode::Off;
  throw std::runtime_error("Unsupported LAMMPS-JAX newton mode '" + value + "'");
}

Precision parse_precision(const std::string &value)
{
  if (value == "float32") return Precision::Float32;
  if (value == "float64") return Precision::Float64;
  throw std::runtime_error("Unsupported LAMMPS-JAX precision '" + value +
                           "'; expected float32 or float64");
}

std::string decode_base64(const std::string &encoded)
{
  static const std::string alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<int> table(256, -1);
  for (int i = 0; i < static_cast<int>(alphabet.size()); ++i)
    table[static_cast<unsigned char>(alphabet[i])] = i;

  std::string out;
  int value = 0;
  int bits = -8;
  for (const unsigned char c : encoded) {
    if (std::isspace(c)) continue;
    if (c == '=') break;
    if (table[c] < 0) throw std::runtime_error("Invalid base64 data");
    value = (value << 6) + table[c];
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<char>((value >> bits) & 0xff));
      bits -= 8;
    }
  }
  return out;
}

// <name>_b64 is base64 VHLO; legacy bundles stored StableHLO text under <name>. PJRT takes both.
std::string get_program(const std::string &json, const std::string &name)
{
  const std::string bytecode_key = name + "_b64";
  if (json.find("\"" + bytecode_key + "\"") != std::string::npos)
    return decode_base64(get_string(json, bytecode_key));
  return get_string(json, name);
}

} // namespace

// Schema comes from export_model in export.py; contract keys map onto ModelContract fields.
ModelBundle load_bundle_file(const std::string &path)
{
  std::ifstream file(path);
  if (!file.is_open()) throw std::runtime_error("Could not open model bundle: " + path);
  const std::string json{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  ModelBundle bundle;
  bundle.format = get_string(json, "format");
  bundle.programs.force_mlir = get_program(json, "force_mlir");
  bundle.programs.energy_mlir = get_program(json, "energy_mlir");
  bundle.programs.energy_and_forces_mlir = get_program(json, "energy_and_forces_mlir");
  bundle.compile_options = decode_base64(get_string(json, "compile_options_b64"));
  // Sorted keys put "contract" before the program blobs; first-match lookup never reads them.
  parse_input_layout(get_string(json, "input_layout"));
  bundle.contract.max_atoms = static_cast<int>(get_number(json, "max_atoms"));
  bundle.contract.max_edges = static_cast<int>(get_number(json, "max_edges"));
  bundle.contract.cutoff = get_number(json, "cutoff");
  bundle.contract.unit_style = get_string(json, "unit_style");
  bundle.contract.precision = parse_precision(get_string(json, "precision"));
  bundle.contract.force_layout = parse_force_layout(get_string(json, "force_output"));
  bundle.contract.newton = parse_newton_mode(get_string(json, "newton"));
  bundle.contract.n_hops = get_int_or(json, "n_hops", 1);
  bundle.contract.comm_widths = get_int_array_or(json, "comm_widths");
  bundle.contract.custom_call_targets = get_string_array_or(json, "custom_call_targets");
  bundle.contract.uses_box = get_bool(json, "uses_box");
  bundle.contract.n_species = get_int_or(json, "n_species", 0);
  if (bundle.contract.n_species < 0) throw std::runtime_error("Invalid n_species in bundle");

  // Distinct tags make loaders that predate n_hops, comm, or half-edge packing reject the bundle.
  if (bundle.format != "lammps-jax-json" &&
      bundle.format != "lammps-jax-json-distributed" &&
      bundle.format != "lammps-jax-json-half-edge")
    throw std::runtime_error("Unsupported bundle format");
  if (bundle.format == "lammps-jax-json-half-edge") {
    const std::string pairing = get_string(json, "edge_pairing");
    if (pairing == "half")
      bundle.contract.half_edges = true;
    else if (pairing != "full")
      throw std::runtime_error("Invalid edge_pairing in bundle");
  }
  if (bundle.contract.half_edges && bundle.contract.n_hops == 1 &&
      bundle.contract.comm_widths.empty())
    throw std::runtime_error(
        "Half-edge bundles require n_hops > 1 or a communication schedule; the "
        "single-hop packer does not deduplicate edge directions");
  if (bundle.contract.max_atoms <= 0 || bundle.contract.max_edges <= 0)
    throw std::runtime_error("Invalid fixed capacities in bundle");
  if (bundle.contract.n_hops < 1) throw std::runtime_error("Invalid n_hops in bundle");
  if (bundle.contract.n_hops > 1 &&
      (bundle.contract.newton != NewtonMode::On || bundle.contract.force_layout != ForceLayout::Atom))
    throw std::runtime_error(
        "Bundles with n_hops > 1 require newton on and atom-force output; ghost force rows "
        "must flow back through the LAMMPS reverse communication");
  for (const int width : bundle.contract.comm_widths)
    if (width <= 0) throw std::runtime_error("Invalid communication width in bundle");
  if (!bundle.contract.comm_widths.empty() &&
      (bundle.contract.newton != NewtonMode::On ||
       bundle.contract.force_layout != ForceLayout::Atom || bundle.contract.n_hops != 1))
    throw std::runtime_error(
        "Communicating bundles require newton on, atom-force output, and a one-cutoff "
        "ghost shell (n_hops = 1)");
  if (!bundle.contract.comm_widths.empty() && bundle.contract.precision == Precision::Float64)
    throw std::runtime_error(
        "float64 communicating bundles are not supported: the in-program feature "
        "exchange path is float32-only. Re-export the model with precision float32 "
        "or without in-program communication");
  return bundle;
}

} // namespace lammps_jax
