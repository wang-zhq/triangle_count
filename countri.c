#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <omp.h>

unsigned long long get_filesize(char const *filename)
{
    struct stat statbuf;
    int r = stat(filename, &statbuf);
    if (r == -1)
    {
        printf("Get size of file is failed!\n");
        exit(-1);
    }
    unsigned long long size = statbuf.st_size;
 
    return size;
}

int comp_ll(const void *a, const void *b)
{
    if (*(unsigned long long *)a < *(unsigned long long *)b)
        return -1;
    else
        return 1;
}

int main(int argc, char const *argv[])
{
    char const *dt_name;
    // 接收命令并读取文件
    if (argc < 1)
    {
        printf("Usage: %s filename.\n", argv[0]);
        exit(1);
    }
    else if (argc == 3)
        dt_name = argv[2];
    else
        dt_name = argv[1];

    unsigned long long dt_size = get_filesize(dt_name);

    printf("The size of data file %s is %lld Bytes.\n", dt_name, dt_size);
    time_t t0 = time(NULL);

    int fp = open(dt_name, O_RDONLY);
    if (fp == -1)
    {
        printf("Can't open %s.\n", dt_name);
        exit(1);
    } 

    unsigned int *endpoints;
    endpoints = mmap(NULL, dt_size, PROT_READ, MAP_SHARED, fp, 0);
    if (endpoints == NULL || endpoints == (void*)-1)
    {
        printf("Mapping Failed!\n");
        close(fp);
        exit(-2);
    }

    unsigned long long n_edge = dt_size/8;
    printf("Data is loaded successfully. There are %lld edges.\n", n_edge);

    long long i;
    unsigned int *front;
    unsigned int *behind;
    front = (unsigned int *)malloc(sizeof(unsigned int)*n_edge);
    behind = (unsigned int *)malloc(sizeof(unsigned int)*n_edge);
    #pragma omp parallel for
    for (i = 0; i < n_edge; i++)
    {
        if (endpoints[2*i] > endpoints[2*i+1])
        {
            front[i] = endpoints[2*i+1];
            behind[i] = endpoints[2*i];
        }
        else
        {
            front[i] = endpoints[2*i];
            behind[i] = endpoints[2*i+1];
        }
    }
    // 关闭镜像通道
    munmap(endpoints, dt_size);
    close(fp);

    unsigned long long dim = 0;
    //#pragma omp parallel for reduction(>:dim)
    for (i = 0; i < n_edge; i++)
        dim = (behind[i] > dim) ? behind[i] : dim;

    dim++;
    printf("The quantity of nodes is %lld.\n", dim);

    unsigned long long * edges;
    edges = (unsigned long long *)malloc(sizeof(unsigned long long)*n_edge);
    #pragma omp parallel for
    for (i = 0; i < n_edge; i++)
        edges[i] = front[i]*dim + behind[i];

    printf("Data preprocessing starts!\n");

    qsort(edges, n_edge, sizeof(unsigned long long), comp_ll);
    printf("Edges sorted!\n");

    char *vld;
    vld = (char *)malloc(sizeof(char)*n_edge);
    #pragma omp parallel for
    for (i = 0; i < n_edge; i++)
    {
        front[i] = edges[i]/dim;
        behind[i] = edges[i]%dim;
        vld[i] = (front[i] == behind[i]) ? 0 : 1;
    }
    free(edges);        

    #pragma omp parallel for
    for (i = 1; i < n_edge; i++)
    {
        if ((front[i] == front[i-1]) && (behind[i] == behind[i-1]))
            vld[i] = 0;
    }
    
    unsigned long long num = 0;
    //#pragma omp parallel for reduction(+:num)
    for (i = 0; i < n_edge; i++)
        num += vld[i];
    
    printf("The number of effective edges is %lld.\n", num);

    unsigned int *row_0;
    unsigned int *col_0;
    row_0 = (unsigned int *)malloc(sizeof(unsigned int)*num);
    col_0 = (unsigned int *)malloc(sizeof(unsigned int)*num);
    long long p_count = 0;
    for (i = 0; i < n_edge; i++)
    {
        if (vld[i] == 1)
        {
            row_0[p_count] = front[i];
            col_0[p_count] = behind[i];
            p_count++;
        }
    }

    free(front);
    free(behind);
    free(vld);

    unsigned int *rows_count;
    rows_count = (unsigned int *)malloc(sizeof(unsigned int)*dim);
    unsigned int *cols_count;
    cols_count = (unsigned int *)malloc(sizeof(unsigned int)*dim);

    #pragma omp parallel for
    for (i = 0; i < dim; i++)
    {
        rows_count[i] = 0;
        cols_count[i] = 0;
    }
        
    // 有累加关系，暂时不并行
    for (i = 0; i < num; i++)
    {
        rows_count[row_0[i]]++; 
        cols_count[col_0[i]]++; 
    }
        
    unsigned long long *rows_addr;
    rows_addr = (unsigned long long *)malloc(sizeof(unsigned long long)*(dim+1));
    unsigned long long *cols_addr;
    cols_addr = (unsigned long long *)malloc(sizeof(unsigned long long)*(dim+1));
    
    rows_addr[0] = 0;
    cols_addr[0] = 0;
    for (i = 1; i <= dim; i++)
    {
        rows_addr[i] = rows_addr[i-1] + rows_count[i-1];
        cols_addr[i] = cols_addr[i-1] + cols_count[i-1];
    }
        
    free(rows_count);
    free(cols_count);

    // 将列进行排序，复用行排序结果
    
    unsigned long long *coloc;
    coloc = (unsigned long long *)malloc(sizeof(unsigned long long)*dim);
    #pragma omp parallel for
    for (i = 0; i < dim; i++)
        coloc[i] = cols_addr[i];

    unsigned int *row_b;
    unsigned int *col_b;
    row_b = (unsigned int *)malloc(sizeof(unsigned int)*num);
    col_b = (unsigned int *)malloc(sizeof(unsigned int)*num);

    for (i = 0; i < num; i++)
    {
        long long k = coloc[col_0[i]];
        col_b[k] = col_0[i];
        row_b[k] = row_0[i];
        coloc[col_0[i]]++;
    }
    free(coloc);
    
    printf("Data preprocessing is completed!\n");

    unsigned long *val_mx;
    val_mx = (unsigned long *)malloc(sizeof(unsigned long)*num);

    #pragma omp parallel for
    for (i = 0; i < num; i++)
    {
        unsigned int rid = row_0[i];
        unsigned int cid = col_0[i];
        
        unsigned long long r_loc = rows_addr[rid];
        unsigned long long c_loc = cols_addr[cid];
        unsigned long long rl_n = rows_addr[rid+1];
        unsigned long long cl_n = cols_addr[cid+1];

        val_mx[i] = 0;
        while ((r_loc < rl_n) && (c_loc < cl_n))
        {
            if (col_0[r_loc] == row_b[c_loc])
            {
                val_mx[i]++;
                r_loc++;
                c_loc++;
            }
            else if (col_0[r_loc] > row_b[c_loc])
                c_loc++;
            else
                r_loc++;
        }

        if (i%dim == 1)
            printf("Counting is processing %6.2f%%\n", (i + 0.0)/num*100.0);
        
    }

    unsigned long long tri_ttl = 0;
    //#pragma omp parallel for reduction(+:tri_ttl)
    for (i = 0; i < num; i++)
        tri_ttl += val_mx[i];

    printf("Time costs %ld sec.\n", time(NULL) - t0 );
    printf("There are \033[1m%lld\033[0m triangles in the input graph.\n", tri_ttl);

    free(col_0);
    free(row_0);
    free(col_b);
    free(row_b);
    free(rows_addr);
    free(cols_addr);
    free(val_mx);
    return 0;
}

