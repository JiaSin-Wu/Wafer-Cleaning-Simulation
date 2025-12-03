#include <bits/stdc++.h>
using namespace std;
#include <chrono>
#include <random>
#include <sstream>
#include <unordered_set>

bool load_csv(const string &path, vector<vector<int>> &A) {
  ifstream f(path);
  if (!f)
    return false;
  string line;
  A.clear();
  while (getline(f, line)) {
    vector<int> row;
    size_t pos = 0;
    while (true) {
      size_t q = line.find(',', pos);
      string tok =
          (q == string::npos) ? line.substr(pos) : line.substr(pos, q - pos);
      if (!tok.empty()) {
        char *endp = nullptr;
        long v = strtol(tok.c_str(), &endp, 10);
        if (endp == tok.c_str())
          return false;
        row.push_back((int)v);
      } else
        row.push_back(0);
      if (q == string::npos)
        break;
      pos = q + 1;
    }
    if (!row.empty())
      A.push_back(move(row));
  }
  return !A.empty();
}

vector<vector<int>> build_matrix(double width_cm, double height_cm,
                                 double res_cm) {
  int W = int(round(width_cm / res_cm)) + 1;
  int H = int(round(height_cm / res_cm)) + 1;
  return vector<vector<int>>(H, vector<int>(W, 0));
}

using Point = pair<int, int>;
struct PointHash {
  size_t operator()(const Point &p) const {
    return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
  }
};

vector<vector<int>> Cleaning(double Wafew_rad, double wafer_resolution_cm,
                             double wafer_rotation, double brush_rotation,
                             double clean_time_sec, double time_resolution,
                             const vector<vector<int>> &brush_topology) {
  if (brush_topology.empty())
    return {};
  auto wafer = build_matrix(2 * Wafew_rad, 2 * Wafew_rad, wafer_resolution_cm);
  int H = wafer.size();
  int W = wafer[0].size();

  for (double t = 0.0; t <= clean_time_sec; t += time_resolution) {
    double wafer_angle = wafer_rotation * t / 60.0 * 360.0;
    int brush_row_count = (int)brush_topology.size();
    int brush_idx =
        (int)(brush_rotation * t / 60.0 * brush_row_count) % brush_row_count;
    const vector<int> &brush_row = brush_topology[brush_idx];

    unordered_set<Point, PointHash> visited_this_round;

    for (double r = -Wafew_rad; r <= Wafew_rad;
         r += 0.5 * wafer_resolution_cm) {
      double ang = wafer_angle * 3.14159265358979323846 / 180.0;
      double y_cm = Wafew_rad + r * sin(ang);
      double x_cm = Wafew_rad + r * cos(ang);

      int iy = (int)llround(y_cm / wafer_resolution_cm);
      int ix = (int)llround(x_cm / wafer_resolution_cm);
      if (iy < 0 || iy >= H || ix < 0 || ix >= W)
        continue;

      Point loc = {iy, ix};
      if (visited_this_round.count(loc))
        continue;

      int B_index = (int)llround((r + Wafew_rad) / (2.0 * Wafew_rad) *
                                 (int)brush_row.size());
      if (B_index < 0)
        B_index = 0;
      if (B_index >= (int)brush_row.size())
        B_index = (int)brush_row.size() - 1;

      wafer[iy][ix] += brush_row[B_index];
      visited_this_round.insert(loc);
    }
  }
  return wafer;
}

vector<uint8_t> make_png_gray8(int W, int H, const vector<uint8_t> &img);
void save_png(const string &path, const vector<vector<int>> &A);

vector<uint8_t> make_png_gray8(int W, int H, const vector<uint8_t> &img) {
  vector<uint8_t> raw;
  for (int j = 0; j < H; ++j) {
    raw.push_back(0); // no filter
    raw.insert(raw.end(), img.begin() + j * W, img.begin() + j * W + W);
  }

  vector<uint8_t> z;
  z.push_back(0x78);
  z.push_back(0x01); // deflate header
  size_t p = 0;
  while (p < raw.size()) {
    size_t n = min<size_t>(65535, raw.size() - p);
    z.push_back((p + n == raw.size()) ? 1 : 0);
    uint16_t len = n, nlen = ~len;
    z.push_back(len & 255);
    z.push_back(len >> 8);
    z.push_back(nlen & 255);
    z.push_back(nlen >> 8);
    z.insert(z.end(), raw.begin() + p, raw.begin() + p + n);
    p += n;
  }

  auto adler32 = [](const uint8_t *d, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) {
      a = (a + d[i]) % 65521;
      b = (b + a) % 65521;
    }
    return (b << 16) | a;
  };

  auto crc32 = [](const uint8_t *d, size_t n) {
    uint32_t c = 0xFFFFFFFF;
    for (size_t i = 0; i < n; ++i) {
      c ^= d[i];
      for (int k = 0; k < 8; ++k)
        c = (c >> 1) ^ (0xEDB88320 * (c & 1));
    }
    return c ^ 0xFFFFFFFF;
  };

  uint32_t ad = adler32(raw.data(), raw.size());
  z.push_back((ad >> 24) & 255);
  z.push_back((ad >> 16) & 255);
  z.push_back((ad >> 8) & 255);
  z.push_back(ad & 255);

  vector<uint8_t> out;
  const uint8_t sig[] = {137, 80, 78, 71, 13, 10, 26, 10};
  out.insert(out.end(), sig, sig + 8);

  auto put_chunk = [&](const char *type, const vector<uint8_t> &data) {
    uint32_t len = data.size();
    out.push_back((len >> 24) & 255);
    out.push_back((len >> 16) & 255);
    out.push_back((len >> 8) & 255);
    out.push_back(len & 255);
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    uint32_t crc =
        crc32(out.data() + out.size() - 4 - data.size(), 4 + data.size());
    out.push_back((crc >> 24) & 255);
    out.push_back((crc >> 16) & 255);
    out.push_back((crc >> 8) & 255);
    out.push_back(crc & 255);
  };

  vector<uint8_t> IHDR = {(uint8_t)(W >> 24),
                          (uint8_t)(W >> 16),
                          (uint8_t)(W >> 8),
                          (uint8_t)(W),
                          (uint8_t)(H >> 24),
                          (uint8_t)(H >> 16),
                          (uint8_t)(H >> 8),
                          (uint8_t)(H),
                          8,
                          0,
                          0,
                          0,
                          0};
  put_chunk("IHDR", IHDR);
  put_chunk("IDAT", z);
  put_chunk("IEND", {});

  return out;
}

void save_png(const string &path, const vector<vector<int>> &A) {
  int H = A.size(), W = A[0].size();
  vector<uint8_t> img(H * W);
  int maxv = 1;
  for (int j = 0; j < H; ++j)
    for (int i = 0; i < W; ++i)
      maxv = max(maxv, A[j][i]);
  for (int j = 0; j < H; ++j) {
    for (int i = 0; i < W; ++i) {
      float scaled = sqrt(A[j][i] / float(maxv)); // √加強暗處
      img[j * W + i] = 255 - uint8_t(255.0 * scaled);
      // img[j * W + i] = 255 - uint8_t(255.0 * A[j][i] / maxv); // 白底黑點
    }
  }
  vector<uint8_t> png = make_png_gray8(W, H, img);
  ofstream f(path, ios::binary);
  f.write((char *)png.data(), (long)png.size());
}

void save_csv(const string &path, const vector<vector<int>> &A) {
  ofstream f(path);
  for (const auto &row : A) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i)
        f << ",";
      f << row[i];
    }
    f << "\n";
  }
}

string build_filename(double Wafew_rad, double wafer_rotation,
                      double brush_rotation) {
  stringstream ss;
  ss << "clean_waf" << int(Wafew_rad) << "_r" << int(wafer_rotation) << "_b"
     << int(brush_rotation) << ".png";
  return ss.str();
}

int main(int argc, char **argv) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Setup for random number generation
  // std::random_device rd;
  // std::mt19937 gen(rd());
  unsigned int seed = 42;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> distrib_rotation(
      30.0, 120.0); // Speeds from 30 to 120 RPM

  vector<vector<int>> brush_topology;
  if (!load_csv("matrix.csv", brush_topology)) {
    cerr << "Error: cannot open or parse matrix.csv\n";
    return 1;
  }

  // parameter
  double wafer_resolution_cm = 0.01; // cm
  double Wafew_rad = 15.0;           // cm
  double clean_time_sec = 60.0;      // sec
  double time_resolution = 0.001;    // sec

  for (int i = 0; i < 10; ++i) {
    // cout << "\n--- Running Simulation " << i + 1 << " of 1000 ---" << endl;
    double wafer_rotation = distrib_rotation(gen);
    double brush_rotation = distrib_rotation(gen);
    auto clean_topolog =
        Cleaning(Wafew_rad, wafer_resolution_cm, wafer_rotation, brush_rotation,
                 clean_time_sec, time_resolution, brush_topology);

    int maxv = 0, nonzero = 0;
    for (auto &row : clean_topolog)
      for (int val : row) {
        if (val > 0)
          nonzero++;
        maxv = max(maxv, val);
      }
    // cout << "Max value = " << maxv << ", Non-zero points = " << nonzero <<
    // endl;

    // New file naming to avoid overwrites
    stringstream ss_prefix;
    ss_prefix << "outputs/run_" << setfill('0') << setw(4) << i + 1
              << "_clean_waf" << int(Wafew_rad) << "_r" << int(wafer_rotation)
              << "_b" << int(brush_rotation);
    string name_prefix = ss_prefix.str();

    string png_file = name_prefix + ".png";
    string csv_file = name_prefix + ".csv";

    // save_png(png_file, clean_topolog);
    save_csv(csv_file, clean_topolog);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  cout << "Total execution time: " << elapsed.count() << "second" << endl;

  return 0;
}
