#include "lammps_jax_model.h"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace lammps_jax {
namespace {

std::string read_text_file(const std::string &path)
{
  std::ifstream file(path);
  if (!file.is_open()) throw std::runtime_error("Could not open model bundle: " + path);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

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

std::string parse_json_string_at(const std::string &json, size_t pos)
{
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

std::string get_string(const std::string &json, const std::string &key)
{
  return parse_json_string_at(json, find_key(json, key));
}

double get_number(const std::string &json, const std::string &key)
{
  size_t pos = find_key(json, key);
  skip_ws(json, pos);
  const size_t end = json.find_first_of(",}\n\r\t ", pos);
  return std::stod(json.substr(pos, end - pos));
}

int get_int(const std::string &json, const std::string &key)
{
  return static_cast<int>(get_number(json, key));
}

bool get_bool(const std::string &json, const std::string &key)
{
  size_t pos = find_key(json, key);
  skip_ws(json, pos);
  if (json.compare(pos, 4, "true") == 0) return true;
  if (json.compare(pos, 5, "false") == 0) return false;
  throw std::runtime_error("Expected boolean for key: " + key);
}

InputLayout parse_input_layout(const std::string &value)
{
  if (value == "sparse-edge") return InputLayout::SparseEdge;
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

} // namespace

ModelBundle load_bundle_file(const std::string &path)
{
  const std::string json = read_text_file(path);
  ModelBundle bundle;
  bundle.format = get_string(json, "format");
  bundle.programs.force_mlir = get_string(json, "force_mlir");
  bundle.programs.energy_mlir = get_string(json, "energy_mlir");
  bundle.programs.energy_and_forces_mlir = get_string(json, "energy_and_forces_mlir");
  bundle.compile_options = decode_base64(get_string(json, "compile_options_b64"));
  // find_key matches the first occurrence in the document; json.dumps with
  // sort_keys=True places "contract" before "metadata", so contract keys win.
  bundle.contract.input_layout = parse_input_layout(get_string(json, "input_layout"));
  bundle.contract.max_atoms = get_int(json, "max_atoms");
  bundle.contract.max_edges = get_int(json, "max_edges");
  bundle.contract.cutoff = get_number(json, "cutoff");
  bundle.contract.unit_style = get_string(json, "unit_style");
  bundle.contract.precision = get_string(json, "precision");
  bundle.contract.force_layout = parse_force_layout(get_string(json, "force_output"));
  bundle.contract.newton = parse_newton_mode(get_string(json, "newton"));
  bundle.contract.uses_box = get_bool(json, "uses_box");

  if (bundle.format != "lammps-jax-json") throw std::runtime_error("Unsupported bundle format");
  if (bundle.contract.precision != "float32") throw std::runtime_error("Only float32 bundles are supported");
  if (bundle.contract.max_atoms <= 0 || bundle.contract.max_edges <= 0)
    throw std::runtime_error("Invalid fixed capacities in bundle");
  return bundle;
}

} // namespace lammps_jax
