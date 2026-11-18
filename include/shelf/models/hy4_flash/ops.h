#pragma once

#include <shelf/models/hy4_flash/residency.h>
#include <shelf/kernels/hy4_flash/linear.h>

namespace shelf::hy4 {
struct Mat {
    uint16_t* p;
    int rows,cols;
    size_t bytes() const {return size_t(rows)*cols*2;}
    Mat row(int i,int count=1) const {return {p+size_t(i)*cols,count,cols};}
};
struct Param {
    const void* p;
    io::SType type;
};

class Ops {
public:
    Residency& gpu;
    int rank;
    Region& arena;
    explicit Ops(Residency& g,int r=0):gpu(g),rank(r),arena(g.workspace[r]){}
    Mat mat(int rows,int cols);
    Param param(const std::string& key) const;
    Mat linear(const std::string& name,Mat input);
    Mat norm(Mat x,const std::string& key,float eps=1e-5f);
    Mat slice(Mat x,int first,int width);
    Mat join(Mat a,Mat b);
    Mat add(Mat a,Mat b);
    Mat swiglu(Mat gate,Mat up,bool clip);
    Mat gather(Mat x,const int* ids,int count);
    Mat embed(const std::vector<int>& ids,bool expand);
    Mat collapse(Mat x,const std::string& prefix,float*& post);
    Mat expand(Mat x,Mat y,const float* post);
    Mat rope(Mat x,int heads,int start,float theta);
    void copy(void* dest,const void* src,size_t bytes,hipMemcpyKind kind);
    int* indices(const std::vector<int>& ids);
};
void checked(hipError_t status);
}
