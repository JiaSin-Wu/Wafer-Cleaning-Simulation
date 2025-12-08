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

// Structure to hold GPU wafer data (keeps data on device for RMS step)
struct GPUWafer {
  int *d_data; // GPU pointer to wafer data
  int H;       // Height
  int W;       // Width
};

// ============================================================================
// Utility Functions
// ============================================================================

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

// ============================================================================
// Step 1. Cleaning Kernel & Function
// ============================================================================

// Optimized Kernel
__global__ void cleaning_kernel(int *d_wafer, int *d_visited_this_round, int H,
                                int W, const int *d_brush_row,
                                int brush_row_size, double Wafew_rad,
                                double wafer_resolution_cm, double sin_val,
                                double cos_val, int current_step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int num_r_steps = (2 * Wafew_rad) / (0.5 * wafer_resolution_cm) + 1;

  if (i >= num_r_steps) {
    return;
  }

  double r = -Wafew_rad + i * (0.5 * wafer_resolution_cm);

  double y_cm = Wafew_rad + r * sin_val;
  double x_cm = Wafew_rad + r * cos_val;

  int iy = (int)llround(y_cm / wafer_resolution_cm);
  int ix = (int)llround(x_cm / wafer_resolution_cm);

  if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
    int old_visited_val =
        atomicExch(&d_visited_this_round[iy * W + ix], current_step);

    if (old_visited_val != current_step) {
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

// Cleaning Function
GPUWafer Cleaning(double Wafew_rad, double wafer_resolution_cm,
                  double wafer_rotation, double brush_rotation,
                  double clean_time_sec, double time_resolution,
                  const vector<vector<int>> &brush_topology) {

  if (brush_topology.empty()) {
    return {nullptr, 0, 0};
  }

  auto wafer = build_matrix(2 * Wafew_rad, 2 * Wafew_rad, wafer_resolution_cm);
  int H = wafer.size();
  int W = wafer[0].size();
  size_t wafer_size_bytes = (size_t)H * W * sizeof(int);

  int *d_wafer;
  cudaMalloc(&d_wafer, wafer_size_bytes);
  cudaMemset(d_wafer, 0, wafer_size_bytes);

  int *d_visited_this_round;
  cudaMalloc(&d_visited_this_round, wafer_size_bytes);
  cudaMemset(d_visited_this_round, -1, wafer_size_bytes);

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

  int num_r_steps = (2 * Wafew_rad) / (0.5 * wafer_resolution_cm) + 1;
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_r_steps + threadsPerBlock - 1) / threadsPerBlock;

  int step_counter = 0;

  for (double t = 0.0; t <= clean_time_sec; t += time_resolution) {
    double wafer_angle = wafer_rotation * t / 60.0 * 360.0;
    double ang_rad = wafer_angle * 3.14159265358979323846 / 180.0;
    double sin_val = sin(ang_rad);
    double cos_val = cos(ang_rad);

    int brush_idx =
        (int)(brush_rotation * t / 60.0 * brush_row_count) % brush_row_count;
    const int *d_brush_row =
        d_brush_topology + (size_t)brush_idx * brush_row_size;

    cleaning_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_wafer, d_visited_this_round, H, W, d_brush_row, brush_row_size,
        Wafew_rad, wafer_resolution_cm, sin_val, cos_val, step_counter);

    step_counter++;
  }
  cudaDeviceSynchronize();

  cudaFree(d_brush_topology);
  cudaFree(d_visited_this_round);

  GPUWafer result;
  result.d_data = d_wafer;
  result.H = H;
  result.W = W;
  return result;
}

// ============================================================================
// Step 2. RMS Evaluation
// ============================================================================

__global__ void compute_sum_kernel(const int *data, int H, int W,
                                   double center_x, double center_y,
                                   double radius_sq, long long *block_sums,
                                   int *block_counts) {
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

__global__ void compute_variance_kernel(const int *data, int H, int W,
                                        double center_x, double center_y,
                                        double radius_sq, double mean,
                                        double *block_sq_devs,
                                        int *block_counts) {
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

// CPU reduction helpers
double reduce_blocks_cpu(double *h_block_data, int num_blocks) {
  double total = 0.0;
  for (int i = 0; i < num_blocks; i++)
    total += h_block_data[i];
  return total;
}
int reduce_blocks_cpu_int(int *h_block_data, int num_blocks) {
  int total = 0;
  for (int i = 0; i < num_blocks; i++)
    total += h_block_data[i];
  return total;
}
long long reduce_blocks_cpu_ll(long long *h_block_data, int num_blocks) {
  long long total = 0;
  for (int i = 0; i < num_blocks; i++)
    total += h_block_data[i];
  return total;
}

// RMS Host Function
double calculate_rms_deviation_circle_gpu(int *d_data, int H, int W,
                                          double wafer_rad,
                                          double resolution_cm) {
  if (H == 0 || W == 0)
    return 0.0;

  int total_pixels = H * W;
  double center_x = wafer_rad / resolution_cm;
  double center_y = wafer_rad / resolution_cm;
  double radius = wafer_rad / resolution_cm;
  double radius_sq = radius * radius;

  int threads_per_block = 512;
  int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

  long long *d_block_sums;
  int *d_block_counts;
  double *d_block_sq_devs;

  cudaMalloc(&d_block_sums, num_blocks * sizeof(long long));
  cudaMalloc(&d_block_counts, num_blocks * sizeof(int));
  cudaMalloc(&d_block_sq_devs, num_blocks * sizeof(double));

  vector<long long> h_block_sums(num_blocks);
  vector<int> h_block_counts(num_blocks);
  vector<double> h_block_sq_devs(num_blocks);

  // PASS 1: Sum and Count
  compute_sum_kernel<<<num_blocks, threads_per_block>>>(
      d_data, H, W, center_x, center_y, radius_sq, d_block_sums,
      d_block_counts);
  cudaDeviceSynchronize();

  cudaMemcpy(h_block_sums.data(), d_block_sums, num_blocks * sizeof(long long),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_block_counts.data(), d_block_counts, num_blocks * sizeof(int),
             cudaMemcpyDeviceToHost);

  long long total_sum = reduce_blocks_cpu_ll(h_block_sums.data(), num_blocks);
  int total_count = reduce_blocks_cpu_int(h_block_counts.data(), num_blocks);

  if (total_count == 0) {
    cudaFree(d_block_sums);
    cudaFree(d_block_counts);
    cudaFree(d_block_sq_devs);
    return 0.0;
  }

  double mean = (double)total_sum / total_count;

  // PASS 2: Variance
  compute_variance_kernel<<<num_blocks, threads_per_block>>>(
      d_data, H, W, center_x, center_y, radius_sq, mean, d_block_sq_devs,
      d_block_counts);
  cudaDeviceSynchronize();

  cudaMemcpy(h_block_sq_devs.data(), d_block_sq_devs,
             num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

  double total_sq_dev = reduce_blocks_cpu(h_block_sq_devs.data(), num_blocks);

  cudaFree(d_block_sums);
  cudaFree(d_block_counts);
  cudaFree(d_block_sq_devs);

  return sqrt(total_sq_dev / total_count);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  auto start_time = high_resolution_clock::now();

  // 1. 設定兩列轉速進行 Enumeration (Total 50 combinations)
  // Wafer Speeds: 5 distinct speeds
  vector<double> wafer_rpms = {30, 40, 50, 60, 70, 75, 80, 90, 100, 110};
  // Brush Speeds: 10 distinct speeds
  vector<double> brush_rpms = {30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

  // Load brush topology
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
  cout << "Running integrated pipeline (Enumeration 50 simulations)...\n";
  cout << "Parameters: wafer_radius=" << Wafer_rad
       << "cm, resolution=" << wafer_resolution_cm << "cm\n\n";

  // Ensure output directory exists (simple check for linux/unix)
  struct stat st = {0};
  if (stat("outputs", &st) == -1) {
#ifdef _WIN32
    _mkdir("outputs");
#else
    mkdir("outputs", 0777);
#endif
  }

  int sim_count = 0;
  int total_sims = wafer_rpms.size() * brush_rpms.size();

  // 2. 雙層迴圈遍歷所有轉速組合
  for (double wafer_rotation : wafer_rpms) {
    for (double brush_rotation : brush_rpms) {
      sim_count++;
      cout << "=== Simulation " << sim_count << " / " << total_sims
           << " ===" << endl;
      cout << "Wafer rotation: " << wafer_rotation
           << " rpm, Brush rotation: " << brush_rotation << " rpm" << endl;

      // STEP 1: Cleaning (Optimized, data stays on GPU)
      auto start_cleaning = high_resolution_clock::now();

      GPUWafer gpu_wafer = Cleaning(
          Wafer_rad, wafer_resolution_cm, wafer_rotation, brush_rotation,
          clean_time_sec, time_resolution, brush_topology);

      auto end_cleaning = high_resolution_clock::now();
      auto duration_cleaning =
          duration_cast<milliseconds>(end_cleaning - start_cleaning);

      cout << "Cleaning time: " << duration_cleaning.count() << " ms" << endl;

      // STEP 2: RMS Calculation (Uses GPU data directly)
      auto start_rms = high_resolution_clock::now();

      double rms_deviation = calculate_rms_deviation_circle_gpu(
          gpu_wafer.d_data, gpu_wafer.H, gpu_wafer.W, Wafer_rad,
          wafer_resolution_cm);

      auto end_rms = high_resolution_clock::now();
      auto duration_rms = duration_cast<microseconds>(end_rms - start_rms);

      cout << "RMS calculation time: " << duration_rms.count()
           << " microseconds" << endl;
      cout << "RMS Deviation: " << rms_deviation << endl;

      // STEP 3: Copy to CPU for CSV Saving
      auto start_csv_prep = high_resolution_clock::now();

      vector<int> flat_wafer(gpu_wafer.H * gpu_wafer.W);
      cudaMemcpy(flat_wafer.data(), gpu_wafer.d_data,
                 gpu_wafer.H * gpu_wafer.W * sizeof(int),
                 cudaMemcpyDeviceToHost);

      vector<vector<int>> wafer_2d(gpu_wafer.H, vector<int>(gpu_wafer.W));
      for (int j = 0; j < gpu_wafer.H; ++j) {
        for (int k = 0; k < gpu_wafer.W; ++k) {
          wafer_2d[j][k] = flat_wafer[j * gpu_wafer.W + k];
        }
      }

      auto end_csv_prep = high_resolution_clock::now();
      // duration_cast<milliseconds>(end_csv_prep - start_csv_prep);

      // Save CSV with specific RPMs in filename
      stringstream ss_prefix;
      ss_prefix << "outputs/run_" << setfill('0') << setw(3) << sim_count
                << "_W" << int(wafer_rotation) << "_B" << int(brush_rotation);

      string csv_file = ss_prefix.str() + ".csv";
      save_csv(csv_file, wafer_2d);
      cout << "Saved to: " << csv_file << endl;

      // STEP 4: Free GPU memory
      cudaFree(gpu_wafer.d_data);
      cout << endl;
    }
  }

  auto end_time = high_resolution_clock::now();
  chrono::duration<double> elapsed = end_time - start_time;

  cout << "\n============================================" << endl;
  cout << "Total execution time: " << elapsed.count() << " seconds" << endl;
  cout << "Average per simulation: " << elapsed.count() / (double)total_sims
       << " seconds" << endl;

  return 0;
}
