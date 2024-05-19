#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include <sstream>
#include <thread>
#include <vector>
#include <unordered_map>

#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include <x86intrin.h>

#include "util.h"
using namespace::std;


int first(uint64_t *line, int size) {
    int bitmap_size =  (size + 63) >> 6;
    for (int i = bitmap_size-1; i >=0; i--) {
        if (line[i] == 0) {
            // printf("%d==0 continue\n",i);
            continue;
        }
        return (i << 6) + 64 - _lzcnt_u64(line[i]) -1;
    }
    return -1;
}

void set(uint64_t * line,int idx) {
    int block_idx = idx / 64; 
    int bit_idx = idx % 64;   
    line[block_idx] |= (1ULL << bit_idx);

}

void linexor(uint64_t * line1,uint64_t * line2, int size) {
    int group_idx =  (size + 63) >> 6;
    for (int i = 0; i < group_idx; i++) {
        line1[i] ^= line2[i];
    }
}

void simd128_linexor(uint64_t * line1,uint64_t * line2, int size) {
    int group_idx =  (size + 63) >> 6;
    int i = 0;
    int pad = 2;
    long count = group_idx - group_idx%pad;
    for (i = 0; i < count; i+=pad) {
        __m128d ma = _mm_load_pd((double*)(line1+i));
        __m128d mb = _mm_load_pd((double*)(line2+i));
        __m128d mc = _mm_xor_pd(ma, mb);
        _mm_store_pd((double*)(line1+i),mc);
    }
    for (;i < group_idx; i++) {
        line1[i] ^= line2[i];
    }
}

void simd512_linexor(uint64_t * line1,uint64_t * line2, int size) {
    int group_idx =  (size + 63) >> 6;
    int i = 0;
    int pad = 8;
    long count = group_idx - group_idx%pad;
    for (i = 0; i < count; i+=pad) {
        __m512i ma = _mm512_loadu_si512 ((line1+i));
        __m512i mb = _mm512_loadu_si512 ((line2+i));
        __m512i mc = _mm512_xor_si512(ma, mb);
        _mm512_storeu_si512((line1+i),mc);
    }
    for (;i < group_idx; i++) {
        line1[i] ^= line2[i];
    }
}

class Grobner {
public:
    Grobner(const std::string &eliminators_file, const std::string &elements_file, long col) {
        col_size = col;
        load_eliminators(eliminators_file);
        load_elements(elements_file);
    }

    void load_eliminators(const std::string &filename) {

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        std::string line;
        int row = 0;
        while (std::getline(file, line)) {
            int size = std::ceil(static_cast<double>(col_size) / 64.0);
            auto item = (uint64_t *)malloc(size*sizeof(uint64_t));
            memset(item, 0, size*sizeof(uint64_t));
            std::istringstream iss(line);
            int index;
            while (iss >> index) {
                set(item,index);
            }
            eliminators.push_back(item);
            row++;
        } 
        file.close();  
    }

    void load_elements(const std::string &filename) {

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        std::string line;
        int row = 0;
        while (std::getline(file, line)) {
            int size = std::ceil(static_cast<double>(col_size) / 64.0);
            auto item = (uint64_t *)malloc(size*sizeof(uint64_t));
            memset(item, 0, size*sizeof(uint64_t));
            std::istringstream iss(line);
            int index;
            while (iss >> index) {
                set(item,index);
            }
            elements.push_back(item);
            row++;
        } 
        file.close();  
    }

    void stroe(const std::string &filename) {
        std::ofstream file(filename, std::ios::out | std::ios::trunc);
        if (!file) {  
            std::cerr << "Unable to open file: " << filename << std::endl;  
            return ;
        }  

        for (const auto &e : elements) {
            for (int j = col_size - 1; j >= 0; --j) {
            int block_idx = j / 64; 
            int bit_idx = j % 64;  

            if (e[block_idx] & (1ULL << bit_idx)) {
                file << j << " "; 
            }
            }
            file << std::endl;
        }

        file.close();
    }

    void start(const std::string & type) {
        auto xorfunc = linexor;
        if(type == "normal") {
            xorfunc = linexor;
        } else if(type == "simd128") {
            xorfunc =  simd128_linexor;
        } else if(type == "simd512") {
            xorfunc =  simd512_linexor;
        } else {
            std::cout << "not  type" << std::endl;
        }
        emap = (linepoint*)malloc(sizeof(linepoint)* (col_size+elements.size()));
        memset(emap, 0, sizeof(linepoint)* (col_size+elements.size()));
        for(auto it:eliminators) {
            auto epos = first(it,col_size);
            emap[epos] = it;
        }

        for(long i=0;i<elements.size();i++) {
            auto ele = elements[i];
            // printf("start\n");
            while(1) {
                auto pos = first(ele,col_size);
                // printf("i:%d pos:%d\n",i,pos);;
                if(pos < 0) { //all zero
                    break;
                }
                if(emap[pos]) {
                    xorfunc(ele, emap[pos], col_size);
                } else {
                    emap[pos] = ele;
                    break;
                }
            }  
        }
    }

    void start_pp(const std::string & type,int pp_number) {
        auto xorfunc = linexor;
        if(type == "normal") {
            xorfunc = linexor;
        } else if(type == "simd128") {
            xorfunc =  simd128_linexor;
        } else if(type == "simd512") {
            xorfunc =  simd512_linexor;
        } else {
            std::cout << "not  type" << std::endl;
        }

        emap = (linepoint*)malloc(sizeof(linepoint)* (col_size+elements.size()));
        memset(emap, 0, sizeof(linepoint)* (col_size+elements.size()));
        for(auto it:eliminators) {
            auto epos = first(it,col_size);
            emap[epos] = it;
        }

        std::thread threads[pp_number];
        for(int i=0;i<pp_number;i++) {
            threads[i] = std::thread([&, idx=i](){
                int id = idx;
                while(id < elements.size()) {
                    auto ele = elements[id];
                    bool flag = true;
                    while(flag) {
                        auto pos = first(ele,col_size);
                        if(pos < 0) { //all zero
                            flag = false;
                            break;
                        }
                        int bid = pos % bucket_count;
                        buckets[bid].rw.readLock();
                        if(emap[pos]) {
                            auto er = emap[pos];
                            buckets[bid].rw.readUnlock();
                            xorfunc(ele, er, col_size);
                        } else {
                            buckets[bid].rw.readUnlock();
                            buckets[bid].rw.writeLock();
                            if(!emap[pos]) {
                                emap[pos] = ele;
                                flag = false;
                            }
                            buckets[bid].rw.writeUnlock();
                        }
                    } 
                    id += pp_number;
                }
            });
        }
        for(int i=0;i<pp_number;i++) {
            threads[i].join();
        }
    }

    vector<uint64_t *> eliminators;     // 消元子
    linepoint* emap;
    // unordered_map<int,uint64_t *> emap;           // 记录首位 串行版
    vector<uint64_t *> elements;        // 被消元行
    long col_size;
    Conmap buckets[32];
    int bucket_count = 32;
};

int main(int argc, char * argv[]) {
    if (argc < 4) {  
        std::cerr << "need test_number and type" << std::endl;  
        return -1;
    }  
    char *end;  
    long number = std::strtol(argv[1], &end, 10) - 1; 
    if (*end != '\0' || number < 0) {  
        std::cerr << "need test_number>0" << std::endl;  
        return -1;
    }  
    const std::string type(argv[2]);

    long pp_number = std::strtol(argv[3], &end, 10); 
    if (*end != '\0' || number < 0) {  
        std::cerr << "need pp_number>0" << std::endl;  
        return -1;
    } 

    vector<std::string> test;
    vector<long> size;
    test.push_back("data/测试样例1 矩阵列数130，非零消元子22，被消元行8");
    size.push_back(130);
    test.push_back("data/测试样例2 矩阵列数254，非零消元子106，被消元行53");
    size.push_back(254);
    test.push_back("data/测试样例3 矩阵列数562，非零消元子170，被消元行53");
    size.push_back(562);
    test.push_back("data/测试样例4 矩阵列数1011，非零消元子539，被消元行263");
    size.push_back(1011);
    test.push_back("data/测试样例5 矩阵列数2362，非零消元子1226，被消元行453");
    size.push_back(2362);
    test.push_back("data/测试样例6 矩阵列数3799，非零消元子2759，被消元行1953");
    size.push_back(3799);
    test.push_back("data/测试样例7 矩阵列数8399，非零消元子6375，被消元行4535");
    size.push_back(8399);
    test.push_back("data/测试样例8 矩阵列数23045，非零消元子18748，被消元行14325");
    size.push_back(23045);
    test.push_back("data/测试样例9 矩阵列数37960，非零消元子29304，被消元行14921");
    size.push_back(37960);
    test.push_back("data/测试样例10 矩阵列数43577，非零消元子39477，被消元行54274");
    size.push_back(43577);
    test.push_back("data/测试样例11 矩阵列数85401，非零消元子5724，被消元行756");
    size.push_back(85401);

    Grobner g(test[number]+"/消元子.txt",test[number]+"/被消元行.txt",size[number]);
    RtimeProfiler tmr(std::to_string(number+1) +" "+ type +" " + std::to_string(pp_number));
    tmr.start();
    if(pp_number > 1)
        g.start_pp(type,pp_number);
    else
        g.start(type);
    
    tmr.end();
    tmr.display_avg();

    g.stroe("result.txt");

    return 0;
}
