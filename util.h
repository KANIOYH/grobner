#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <iostream>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include <x86intrin.h>

typedef std::chrono::steady_clock::time_point ctime_t;

void printBits(uint64_t const value) {  
    for (int i = 63; i >= 0; --i) {  
        std::cout << ((value >> i) & 1);  
    }  
    std::cout <<" ";

}

void test_simd() {
    uint64_t a[] = {1,127};
    printf("a:\n");
    printBits(a[0]);
    printBits(a[1]);
    printf("\n");
    uint64_t b[] = {2,127};
    printf("b:\n");
    printBits(b[0]);
    printBits(b[1]);
    printf("\n");
    uint64_t c[] = {0,0};
    printf("c:\n");
    printBits(c[0]);
    printBits(c[1]);
    printf("\n");
    __m128d ma = _mm_load_pd((double*)a);
    __m128d mb = _mm_load_pd((double*)b);
    __m128d mc = _mm_xor_pd(ma, mb);
    _mm_store_pd((double*)c,mc);
    printf("result c:\n");
    printBits(c[0]);
    printBits(c[1]);
    printf("\n");
}

class RWLock {
private:
    std::mutex readMtx;
    std::mutex writeMtx;
    int readCnt; // 已加读锁个数
public:
    RWLock() : readCnt(0) {}
    void readLock()
    {
        readMtx.lock();
        if (++readCnt == 1) {
            writeMtx.lock();  // 存在线程读操作时，写加锁（只加一次）
        }
        readMtx.unlock();
    }
    void readUnlock()
    {
        readMtx.lock();
        if (--readCnt == 0) { // 没有线程读操作时，释放写锁
            writeMtx.unlock();
        }
        readMtx.unlock();
    }
    void writeLock()
    {
        writeMtx.lock();
    }
    void writeUnlock()
    {
        writeMtx.unlock();
    }
};

class RtimeProfiler {

public:
    RtimeProfiler(std::string name):name_(name) {};
    ~RtimeProfiler() {};

    void start() {
        start_times.push_back(std::chrono::steady_clock::now());
    };

    void end() {
        end_times.push_back(std::chrono::steady_clock::now());
    };

    void display_avg() {
        std::vector<double> times;
        for(int i = 0; i < start_times.size(); i++) {
            double time = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_times[i] - start_times[i]).count()/1000;
            times.push_back(time);
        }
        double sumValue = std::accumulate(times.begin(), times.end(),0.0);
        printf("%s %lf\n",name_.c_str(),sumValue/times.size());       
    }

private:
    std::vector<ctime_t> start_times;
    std::vector<ctime_t> end_times;
    std::string name_;
};