/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号:SA24221035
 * 姓名:商坤杰
 * 邮箱:sun_kj@mail.ustc.edu.cn
 ------------------------------------------------*/

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]

using std::cin, std::cout, std::endl;
using std::ifstream, std::ofstream;

// 存活细胞数
int population(int N, char *universe)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

// 打印世界状态
void print_universe(int N, char *universe)
{
    // 仅在N较小(<= 32)时用于Debug
    if (N > 32)
        return;
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int z = 0; z < N; z++)
            {
                if (AT(x, y, z))
                    cout << "O ";
                else
                    cout << "* ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "population: " << population(N, universe) << endl;
}

// kernel代码,计算下一状态
__global__ void life3d_kernel(unsigned char* current, unsigned char* next, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (x>=N || y>N || z>=N) return; //N与THREADS_PER_DIM非整数倍情况会产生冗余线程的情况
	
	int idx = x*N*N + y*N + z;
	
	int alive = 0;
	for (int dx = -1; dx <= 1; dx++)
		for (int dy = -1; dy <= 1; dy++)
			for (int dz = -1; dz <= 1; dz++)
			{
				if (dx == 0 && dy == 0 && dz == 0)
					continue;
				int nx = (x + dx + N) % N;
				int ny = (y + dy + N) % N;
				int nz = (z + dz + N) % N;
				int n_idx = nx*N*N + ny*N + nz;
				alive += current[n_idx];
			}
	if (current[idx] && (alive < 5 || alive > 7))
		next[idx] = 0;
	else if (!current[idx] && alive == 6)
		next[idx] = 1;
	else
		next[idx] = current[idx];
}
// CUDA实现将世界推进T个时刻
void life3d_run_cuda(int N, char *universe, int T){
	size_t size = N*N*N*sizeof(unsigned char);
	
	unsigned char *d_current,*d_next;
	
	cudaMalloc((void**)&d_current,size);
	cudaMalloc((void**)&d_next,size);
	cudaMemcpy(d_current,universe,size,cudaMemcpyHostToDevice);
	
	int THREAD_PER_DIM = 8;
	dim3 threads(THREAD_PER_DIM,THREAD_PER_DIM,THREAD_PER_DIM);
	dim3 blocks((N-1)/threads.x + 1,(N-1)/threads.y + 1,(N-1)/threads.z + 1);
	
	for(int t = 0;t < T; t++){
		life3d_kernel<<<blocks,threads>>>(d_current,d_next,N); //计算每轮结果
		std::swap(d_current,d_next);
	}
	
	cudaDeviceSynchronize;	//同步
	
	cudaMemcpy(universe, d_current, size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_current);
    cudaFree(d_next);
}

// 读取输入文件
void read_file(char *input_file, char *buffer)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size))
    {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        cout << "Error: Could not open file " << output_file << std::endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char **argv)
{
    // cmd args
    if (argc < 5)
    {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];

    char *universe = (char *)malloc(N * N * N);
    read_file(input_file, universe);

    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();
    life3d_run_cuda(N, universe, T);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);
    write_file(output_file, universe, N);

    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;

    free(universe);
    return 0;
}
