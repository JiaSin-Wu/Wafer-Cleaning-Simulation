#include <bits/stdc++.h>
#include <chrono>
#include <random>
#include <sstream>
#include <sys/stat.h> // For mkdir

// CUDA specific includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// --- Utility Functions (largely unchanged from CPU version) ---

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

// --- CUDA Kernel ---

__global__ void cleaning_kernel(int *d_wafer, int *d_visited_this_round, int H,
                                int W, const int *d_brush_row,
                                int brush_row_size, double Wafew_rad,
                                double wafer_resolution_cm, double sin_ang,
                                double cos_ang) {
  // Calculate the global thread ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate total number of steps for 'r'
  int num_r_steps = (2 * Wafew_rad) / (0.5 * wafer_resolution_cm) + 1;

  // Boundary check for threads
  if (i >= num_r_steps) {
    return;
  }

  // Map thread ID to a radial position 'r'
  double r = -Wafew_rad + i * (0.5 * wafer_resolution_cm);

  // Perform coordinate calculations using pre-calculated sin/cos
  double y_cm = Wafew_rad + r * sin_ang;
  double x_cm = Wafew_rad + r * cos_ang;

  int iy = (int)llround(y_cm / wafer_resolution_cm);
  int ix = (int)llround(x_cm / wafer_resolution_cm);

  // Check if the calculated coordinates are within the wafer bounds
  if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
    // Atomically "claim" the cell for this time step.
    // atomicExch returns the OLD value at the location.
    int old_visited_val = atomicExch(&d_visited_this_round[iy * W + ix], 1);

    // If the old value was 0, this is the first thread to visit this cell in
    // this time step.
    if (old_visited_val == 0) {
      int B_index =
          (int)llround((r + Wafew_rad) / (2.0 * Wafew_rad) * brush_row_size);
      if (B_index < 0)
        B_index = 0;
      if (B_index >= brush_row_size)
        B_index = brush_row_size - 1;

      int brush_value = d_brush_row[B_index];

      // The wafer value accumulates over all time steps, so we still use
      // atomicAdd to be safe, though a simple add would likely work as well
      // since d_wafer is not modified by other kernels concurrently.
      if (brush_value > 0) {
        atomicAdd(&d_wafer[iy * W + ix], brush_value);
      }
    }
  }
}

// --- Main 'Cleaning' function (Host-side) ---

vector<vector<int>> Cleaning(double Wafew_rad, double wafer_resolution_cm,
                             double wafer_rotation, double brush_rotation,
                             double clean_time_sec, double time_resolution,
                             const vector<vector<int>> &brush_topology) {
  if (brush_topology.empty())
    return {};

  // 1. Initialize Wafer Matrix on Host and get dimensions
  auto wafer = build_matrix(2 * Wafew_rad, 2 * Wafew_rad, wafer_resolution_cm);
  int H = wafer.size();
  int W = wafer[0].size();
  size_t wafer_size_bytes = (size_t)H * W * sizeof(int);

  // 2. Allocate GPU memory
  int *d_wafer;
  cudaMalloc(&d_wafer, wafer_size_bytes);
  cudaMemset(d_wafer, 0, wafer_size_bytes);

  int *d_visited_this_round; // Buffer for the "visited" flag
  cudaMalloc(&d_visited_this_round, wafer_size_bytes);

  int *d_brush_topology;
  int brush_row_count = (int)brush_topology.size();
  int brush_row_size = (int)brush_topology[0].size();
  size_t brush_topo_size_bytes =
      (size_t)brush_row_count * brush_row_size * sizeof(int);
  cudaMalloc(&d_brush_topology, brush_topo_size_bytes);

  vector<int> flat_brush_topology;
  flat_brush_topology.reserve(brush_row_count * brush_row_size);
  for (const auto &row : brush_topology) {
    flat_brush_topology.insert(flat_brush_topology.end(), row.begin(),
                               row.end());
  }
  cudaMemcpy(d_brush_topology, flat_brush_topology.data(),
             brush_topo_size_bytes, cudaMemcpyHostToDevice);

  // 3. Configure Kernel launch parameters
  int num_r_steps = (2 * Wafew_rad) / (0.5 * wafer_resolution_cm) + 1;
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_r_steps + threadsPerBlock - 1) / threadsPerBlock;

  // --- CUDA Graph Setup (Manual Node Creation) ---
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaGraphCreate(&graph, 0);

  // Use a non-default stream for graph launch
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Initial (dummy) parameters for node creation
  double sin_ang = 0.0;
  double cos_ang = 1.0;
  const int *d_brush_row = d_brush_topology;

  // Node 1: Memset
  cudaGraphNode_t memsetNode;
  cudaMemsetParams memsetParams = {};
  memsetParams.dst = d_visited_this_round;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(int);
  memsetParams.width = W;
  memsetParams.height = H;
  cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

  // Node 2: Kernel Launch
  cudaGraphNode_t kernelNode;
  cudaKernelNodeParams kernelParams = {};
  void *kernelArgs[] = {(void *)&d_wafer,     (void *)&d_visited_this_round,
                        (void *)&H,           (void *)&W,
                        (void *)&d_brush_row, (void *)&brush_row_size,
                        (void *)&Wafew_rad,   (void *)&wafer_resolution_cm,
                        (void *)&sin_ang,     (void *)&cos_ang};
  kernelParams.func = (void *)cleaning_kernel;
  kernelParams.gridDim = blocksPerGrid;
  kernelParams.blockDim = threadsPerBlock;
  kernelParams.sharedMemBytes = 0;
  kernelParams.kernelParams = kernelArgs;
  kernelParams.extra = NULL;

  // Set dependency: kernelNode depends on memsetNode
  cudaGraphAddKernelNode(&kernelNode, graph, &memsetNode, 1, &kernelParams);

  // Instantiate the graph
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

  // 4. Time-stepping loop (on CPU, using CUDA Graph)
  for (double t = 0.0; t <= clean_time_sec; t += time_resolution) {
    double wafer_angle_rad = wafer_rotation * t / 60.0 * 360.0 * (M_PI / 180.0);
    sin_ang = sin(wafer_angle_rad);
    cos_ang = cos(wafer_angle_rad);

    int brush_idx =
        (int)(brush_rotation * t / 60.0 * brush_row_count) % brush_row_count;
    d_brush_row = d_brush_topology + (size_t)brush_idx * brush_row_size;

    // Update kernel parameters for the current time step
    kernelParams.kernelParams[4] = &d_brush_row;
    kernelParams.kernelParams[8] = &sin_ang;
    kernelParams.kernelParams[9] = &cos_ang;
    cudaGraphExecKernelNodeSetParams(instance, kernelNode, &kernelParams);

    // Launch the graph on the stream
    cudaGraphLaunch(instance, stream);
  }
  cudaStreamSynchronize(stream);

  // 5. Copy results back from GPU to CPU
  vector<int> flat_wafer(H * W);
  cudaMemcpy(flat_wafer.data(), d_wafer, wafer_size_bytes,
             cudaMemcpyDeviceToHost);

  // 6. Free GPU memory and graph resources
  cudaFree(d_wafer);
  cudaFree(d_brush_topology);
  cudaFree(d_visited_this_round);
  cudaGraphExecDestroy(instance);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  // 7. Un-flatten wafer matrix on host
  for (int j = 0; j < H; ++j) {
    for (int i = 0; i < W; ++i) {
      wafer[j][i] = flat_wafer[j * W + i];
    }
  }

  return wafer;
}

// --- PNG and CSV Saving Functions (unchanged) ---

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
      for (int k = 0; k < 8; k++)
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
      float scaled = sqrt(A[j][i] / float(maxv));
      img[j * W + i] = 255 - uint8_t(255.0 * scaled);
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

// --- Main Function (updated for GPU execution) ---

int main(int argc, char **argv) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // std::random_device rd;
  // std::mt19937 gen(rd());
  unsigned int seed = 42;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> distrib_rotation(30.0, 120.0);

  vector<vector<int>> brush_topology;
  if (!load_csv("matrix.csv", brush_topology)) {
    cerr << "Error: cannot open or parse matrix.csv\n";
    return 1;
  }

  double wafer_resolution_cm = 0.01;
  double Wafew_rad = 15.0;
  double clean_time_sec = 60.0;
  double time_resolution = 0.001;

  for (int i = 0; i < 10; ++i) {
    // cout << "\n--- Running GPU Simulation " << i + 1 << " ---" << endl;

    double wafer_rotation = distrib_rotation(gen);
    double brush_rotation = distrib_rotation(gen);

    // cout << "Parameters: wafer_rotation=" << wafer_rotation
    //      << ", brush_rotation=" << brush_rotation << endl;

    auto clean_topolog =
        Cleaning(Wafew_rad, wafer_resolution_cm, wafer_rotation, brush_rotation,
                 clean_time_sec, time_resolution, brush_topology);

    stringstream ss_prefix;
    ss_prefix << "outputs/run_" << setfill('0') << setw(4) << i + 1
              << "_clean_waf" << int(Wafew_rad) << "_r" << int(wafer_rotation)
              << "_b" << int(brush_rotation);

    string png_file = ss_prefix.str() + ".png";
    string csv_file = ss_prefix.str() + ".csv";

    // cout << "Saving results to " << png_file << " and " << csv_file << endl;
    // save_png(png_file, clean_topolog);
    save_csv(csv_file, clean_topolog);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  // cout << "\nTotal GPU execution time: " << elapsed.count() << " seconds"
  //      << endl;

  return 0;
}
