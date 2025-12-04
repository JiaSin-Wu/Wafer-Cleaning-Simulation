#include <bits/stdc++.h>
#include <chrono>
#include <random>
#include <sstream>
#include <sys/stat.h>

// CUDA specific includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace std::chrono;

// Structure to hold GPU wafer data
struct GPUWafer {
  int* d_data;    // GPU pointer to wafer data
  int H;          // Height
  int W;          // Width
};

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

// ============================================================================
// Step 1. Cleaning
// ============================================================================
// --- CUDA Kernel ---
__global__ void cleaning_kernel(int *d_wafer, int *d_visited_this_round, int H,
                                int W, const int *d_brush_row,
                                int brush_row_size, double Wafew_rad,
                                double wafer_resolution_cm,
                                double wafer_angle) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int num_r_steps = (2 * Wafew_rad) / (0.5 * wafer_resolution_cm) + 1;

  if (i >= num_r_steps) {
    return;
  }

  double r = -Wafew_rad + i * (0.5 * wafer_resolution_cm);

  double ang = wafer_angle * 3.14159265358979323846 / 180.0;
  double y_cm = Wafew_rad + r * sin(ang);
  double x_cm = Wafew_rad + r * cos(ang);

  int iy = (int)llround(y_cm / wafer_resolution_cm);
  int ix = (int)llround(x_cm / wafer_resolution_cm);

  if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
    int old_visited_val = atomicExch(&d_visited_this_round[iy * W + ix], 1);

    if (old_visited_val == 0) {
      int B_index =
          (int)llround((r + Wafew_rad) / (2.0 * Wafew_rad) * brush_row_size);
      if (B_index < 0)
        B_index = 0;
      if (B_index >= brush_row_size)
        B_index = brush_row_size - 1;

      int brush_value = d_brush_row[B_index];

      if (brush_value > 0) {
        atomicAdd(&d_wafer[iy * W + ix], brush_value);
      }
    }
  }
}

// Modified 'Cleaning' function (to leave data on GPU for RMS calculation) 
GPUWafer Cleaning(double Wafew_rad, double wafer_resolution_cm,
                  double wafer_rotation, double brush_rotation,
                  double clean_time_sec, double time_resolution,
                  const vector<vector<int>> &brush_topology) {

    if (brush_topology.empty()) {
    GPUWafer empty_result;
    empty_result.d_data = nullptr;
    empty_result.H = 0;
    empty_result.W = 0;
    return empty_result;
    }

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

    // 4. Time-stepping loop (on CPU)
    for (double t = 0.0; t <= clean_time_sec; t += time_resolution) {
        cudaMemset(d_visited_this_round, 0, wafer_size_bytes);

        double wafer_angle = wafer_rotation * t / 60.0 * 360.0;
        int brush_idx =
            (int)(brush_rotation * t / 60.0 * brush_row_count) % brush_row_count;
        const int *d_brush_row =
            d_brush_topology + (size_t)brush_idx * brush_row_size;

        cleaning_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_wafer, d_visited_this_round, H, W, d_brush_row, brush_row_size,
            Wafew_rad, wafer_resolution_cm, wafer_angle);
    }
    cudaDeviceSynchronize();

    // ========== Original code that copied back to CPU ==========
    // vector<int> flat_wafer(H * W);
    // cudaMemcpy(flat_wafer.data(), d_wafer, wafer_size_bytes,
    //            cudaMemcpyDeviceToHost);
    //
    // for (int j = 0; j < H; ++j) {
    //   for (int i = 0; i < W; ++i) {
    //     wafer[j][i] = flat_wafer[j * W + i];
    //   }
    // }
    // cudaFree(d_wafer);  // DON'T free yet!
    // ====================

    // 6. Free GPU memory
    cudaFree(d_brush_topology);
    cudaFree(d_visited_this_round);

    // Add return GPU pointer instead of CPU data
    GPUWafer result;
    result.d_data = d_wafer;  // Keep data on GPU!
    result.H = H;
    result.W = W;
    return result;
}

// ============================================================================
// Step 2. Evaluation
// ============================================================================
// RMS deviation kernels
__global__ void compute_sum_kernel(
  const int* data, int H, int W,
  double center_x, double center_y, double radius_sq,
  long long* block_sums, int* block_counts)
{
  __shared__ long long s_sum[512];
  __shared__ int s_count[512];
    
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  long long my_sum = 0;
  int my_count = 0;
  
  if (idx < H * W) {
      int i = idx % W;
      int j = idx / W;
      
      double dx = i - center_x;
      double dy = j - center_y;
      double dist_sq = dx * dx + dy * dy;

      if (dist_sq <= radius_sq) {
          my_sum = data[idx];
          my_count = 1;
      }
  }
  
  s_sum[tid] = my_sum;
  s_count[tid] = my_count;
  __syncthreads();
  
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
          s_sum[tid] += s_sum[tid + stride];
          s_count[tid] += s_count[tid + stride];
      }
      __syncthreads();
  }
  
  if (tid == 0) {
      block_sums[blockIdx.x] = s_sum[0];
      block_counts[blockIdx.x] = s_count[0];
  }
}

__global__ void compute_variance_kernel(
  const int* data, int H, int W,
  double center_x, double center_y, double radius_sq,
  double mean, double* block_sq_devs, int* block_counts)
{
    __shared__ double s_sq_dev[512];
    __shared__ int s_count[512];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_sq_dev = 0.0;
    int my_count = 0;
    
    if (idx < H * W) {
        int i = idx % W;
        int j = idx / W;
        
        double dx = i - center_x;
        double dy = j - center_y;
        double dist_sq = dx * dx + dy * dy;
        
        if (dist_sq <= radius_sq) {
            double val = (double)data[idx];
            double deviation = val - mean;
            my_sq_dev = deviation * deviation;
            my_count = 1;
        }
    }
    
    s_sq_dev[tid] = my_sq_dev;
    s_count[tid] = my_count;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq_dev[tid] += s_sq_dev[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_sq_devs[blockIdx.x] = s_sq_dev[0];
        block_counts[blockIdx.x] = s_count[0];
    }
}

// CPU reduction functions
double reduce_blocks_cpu(double* h_block_data, int num_blocks) {
    double total = 0.0;
    for (int i = 0; i < num_blocks; i++) {
        total += h_block_data[i];
    }
    return total;
}

int reduce_blocks_cpu_int(int* h_block_data, int num_blocks) {
    int total = 0;
    for (int i = 0; i < num_blocks; i++) {
        total += h_block_data[i];
    }
    return total;
}

long long reduce_blocks_cpu_ll(long long* h_block_data, int num_blocks) {
    long long total = 0;
    for (int i = 0; i < num_blocks; i++) {
        total += h_block_data[i];
    }
    return total;
}

// RMS calculation (now accepts GPU pointer directly)
double calculate_rms_deviation_circle_gpu(
  int* d_data,  // [CHANGED] GPU pointer instead of CPU vector
  int H,        // [ADDED] Height parameter
  int W,        // [ADDED] Width parameter
  double wafer_rad, 
  double resolution_cm)
{
    
    if(H == 0 || W == 0) return 0.0;
    
    int total_pixels = H * W;
    
    double center_x = wafer_rad / resolution_cm;
    double center_y = wafer_rad / resolution_cm;
    double radius = wafer_rad / resolution_cm;
    double radius_sq = radius * radius;
    
    // ========== Flattening code (data already flat on GPU) ==========
    // vector<int> flat_data(total_pixels);
    // for (int j = 0; j < H; j++) {
    //     for (int i = 0; i < W; i++) {
    //         flat_data[j * W + i] = A[j][i];
    //     }
    // }
    // ====================
    
    // ========== Memory allocation and transfer (already on GPU) ==========
    // int* d_data;
    // CUDA_CHECK(cudaMalloc(&d_data, total_pixels * sizeof(int)));
    // CUDA_CHECK(cudaMemcpy(d_data, flat_data.data(), 
    //                      total_pixels * sizeof(int), cudaMemcpyHostToDevice));
    // ====================
    
    // Calculate grid dimensions
    int threads_per_block = 512;
    int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;
    
    // Allocate memory for block results
    long long* d_block_sums;
    int* d_block_counts;
    double* d_block_sq_devs;
    
    cudaMalloc(&d_block_sums, num_blocks * sizeof(long long));
    cudaMalloc(&d_block_counts, num_blocks * sizeof(int));
    cudaMalloc(&d_block_sq_devs, num_blocks * sizeof(double));
    
    // Host memory for block results
    vector<long long> h_block_sums(num_blocks);
    vector<int> h_block_counts(num_blocks);
    vector<double> h_block_sq_devs(num_blocks);
    
    // PASS 1: Calculate sum and count
    compute_sum_kernel<<<num_blocks, threads_per_block>>>(
        d_data, H, W, center_x, center_y, radius_sq,
        d_block_sums, d_block_counts
    );
    cudaGetLastError();
    cudaDeviceSynchronize();
    
    // Copy block results to host
    cudaMemcpy(h_block_sums.data(), d_block_sums, num_blocks * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block_counts.data(), d_block_counts, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Reduce on CPU
    long long total_sum = reduce_blocks_cpu_ll(h_block_sums.data(), num_blocks);
    int total_count = reduce_blocks_cpu_int(h_block_counts.data(), num_blocks);
    
    if (total_count == 0) {
        // [REMOVED] cudaFree(d_data);  // Caller will free this!
        cudaFree(d_block_sums);
        cudaFree(d_block_counts);
        cudaFree(d_block_sq_devs);
        return 0.0;
    }
    
    double mean = (double)total_sum / total_count;
    
    // PASS 2: Calculate variance
    compute_variance_kernel<<<num_blocks, threads_per_block>>>(
        d_data, H, W, center_x, center_y, radius_sq,
        mean, d_block_sq_devs, d_block_counts
    );
    cudaGetLastError();
    cudaDeviceSynchronize();
    
    // Copy variance results to host
    cudaMemcpy(h_block_sq_devs.data(), d_block_sq_devs, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Reduce on CPU
    double total_sq_dev = reduce_blocks_cpu(h_block_sq_devs.data(), num_blocks);
    
    // Cleanup temporary buffers
    // cudaFree(d_data);  // Caller will free this!
    cudaFree(d_block_sums);
    cudaFree(d_block_counts);
    cudaFree(d_block_sq_devs);
    
    // Calculate RMS deviation
    double rms_dev = sqrt(total_sq_dev / total_count);
    return rms_dev;
}

// --- PNG and CSV Saving Functions (unchanged) ---
vector<uint8_t> make_png_gray8(int W, int H, const vector<uint8_t> &img) {
  vector<uint8_t> raw;
  for (int j = 0; j < H; ++j) {
    raw.push_back(0);
    raw.insert(raw.end(), img.begin() + j * W, img.begin() + j * W + W);
  }

  vector<uint8_t> z;
  z.push_back(0x78);
  z.push_back(0x01);
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

int main(int argc, char **argv) {
  auto start_time = high_resolution_clock::now();

  unsigned int seed = 42;
  mt19937 gen(seed);
  uniform_real_distribution<> distrib_rotation(30.0, 120.0);

  // Load brush topology (only CSV input needed)
  vector<vector<int>> brush_topology;
  if (!load_csv("matrix.csv", brush_topology)) {
    cerr << "Error: cannot open or parse matrix.csv\n";
    return 1;
  }

  double wafer_resolution_cm = 0.01;
  double Wafer_rad = 15.0;
  double clean_time_sec = 60.0;
  double time_resolution = 0.001;

  cout << fixed << setprecision(4);
  cout << "Running integrated pipeline (Cleaning + RMS calculation)...\n";
  cout << "Parameters: wafer_radius=" << Wafer_rad 
       << "cm, resolution=" << wafer_resolution_cm << "cm\n\n";

  for (int i = 0; i < 10; ++i) {
    cout << "=== Simulation " << i + 1 << " ===" << endl;

    double wafer_rotation = distrib_rotation(gen);
    double brush_rotation = distrib_rotation(gen);

    cout << "Wafer rotation: " << wafer_rotation 
         << " rpm, Brush rotation: " << brush_rotation << " rpm" << endl;

    // STEP 1: Cleaning wafer (stays on GPU)
    auto start_cleaning = high_resolution_clock::now();
    
    GPUWafer gpu_wafer = Cleaning(  // Returns GPUWafer, not vector
        Wafer_rad, wafer_resolution_cm, wafer_rotation, brush_rotation,
        clean_time_sec, time_resolution, brush_topology);
    
    auto end_cleaning = high_resolution_clock::now();
    auto duration_cleaning = duration_cast<milliseconds>(end_cleaning - start_cleaning);
    
    cout << "Cleaning time: " << duration_cleaning.count() << " ms" << endl;

    // STEP 2: Calculate RMS (data laready on GPU)
    auto start_rms = high_resolution_clock::now();
    
    double rms_deviation = calculate_rms_deviation_circle_gpu(
        gpu_wafer.d_data,  // [CHANGED] Pass GPU pointer directly
        gpu_wafer.H,       // [ADDED] Pass dimensions
        gpu_wafer.W,       // [ADDED] Pass dimensions
        Wafer_rad, 
        wafer_resolution_cm);
    
    auto end_rms = high_resolution_clock::now();
    auto duration_rms = duration_cast<microseconds>(end_rms - start_rms);
    //auto duration_rms = duration_cast<std::chrono::microseconds>(end_rms - start_rms);
    
    cout << "RMS calculation time: " << duration_rms.count() << " microseconds" << endl;
    cout << "RMS Deviation: " << rms_deviation << endl;

    // STEP 3. Copy to CPU ONLY for CSV saving
    auto start_csv_prep = high_resolution_clock::now();
    
    // Copy GPU data to CPU
    vector<int> flat_wafer(gpu_wafer.H * gpu_wafer.W);
    cudaMemcpy(flat_wafer.data(), gpu_wafer.d_data,
                         gpu_wafer.H * gpu_wafer.W * sizeof(int),
                         cudaMemcpyDeviceToHost);
    
    // Convert to 2D for CSV
    vector<vector<int>> wafer_2d(gpu_wafer.H, vector<int>(gpu_wafer.W));
    for (int j = 0; j < gpu_wafer.H; ++j) {
        for (int i = 0; i < gpu_wafer.W; ++i) {
            wafer_2d[j][i] = flat_wafer[j * gpu_wafer.W + i];
        }
    }
    
    auto end_csv_prep = high_resolution_clock::now();
    auto duration_csv_prep = duration_cast<milliseconds>(end_csv_prep - start_csv_prep);
    cout << "CSV preparation time: " << duration_csv_prep.count() << " ms" << endl;
    
    // Save CSV
    stringstream ss_prefix;
    ss_prefix << "outputs/run_" << setfill('0') << setw(4) << i + 1
              << "_clean_waf" << int(Wafer_rad) << "_r" << int(wafer_rotation)
              << "_b" << int(brush_rotation);

    string csv_file = ss_prefix.str() + ".csv";
    save_csv(csv_file, wafer_2d);
    
    cout << "Saved to: " << csv_file << endl;
    
    // STEP 4: Free GPU memory
    cudaFree(gpu_wafer.d_data);
    cout << endl;
  }

  auto end_time = high_resolution_clock::now();
  chrono::duration<double> elapsed = end_time - start_time;

  cout << "\n============================================" << endl;
  cout << "Total execution time: " << elapsed.count() << " seconds" << endl;
  cout << "Average per simulation: " << elapsed.count() / 10.0 << " seconds" << endl;

  return 0;
}