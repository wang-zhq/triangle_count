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

// 据说是给快速索引用的
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
    unsigned int dim = 0;
    for (i = 0; i < n_edge*2; i++)
        dim = (endpoints[i] > dim) ? endpoints[i] : dim;

    dim++;

    printf("The quantity of nodes is %d.\n", dim);

    unsigned long bigmod = dim;
    //printf("bigmod is %ld\n", bigmod);
    unsigned long long *edge_list; 
    edge_list = (unsigned long long *)malloc(sizeof(unsigned long long)*n_edge);

    #pragma omp parallel for
    for (i = 0; i < n_edge; i++)
    {
        unsigned int a = endpoints[2*i];
        unsigned int b = endpoints[2*i+1];
        if (a < b)
            edge_list[i] = a*bigmod + b;

        else if (b < a)
            edge_list[i] = b*bigmod + a;

        else
            edge_list[i] = 0;

    }

    munmap(endpoints, dt_size);
    close(fp);

    printf("Data preprocessing starts!\n");
    qsort(edge_list, n_edge, sizeof(unsigned long long), comp_ll);

    unsigned long *front;
    unsigned long *behind;
    front = (unsigned long *)malloc(sizeof(unsigned long)*n_edge);
    behind = (unsigned long *)malloc(sizeof(unsigned long)*n_edge);

    unsigned long long num;
    if (edge_list[0] > 0)
    {
        front[0]  = edge_list[0] / bigmod;
        behind[0] = edge_list[0] % bigmod;
        num = 1;
    }
    else
        num = 0;
    
    //#pragma omp parallel for reduction(+:num)
    for (i = 1; i < n_edge; i++)
    {
        if (edge_list[i] > edge_list[i-1])
        {
            front[num]  = edge_list[i] / bigmod;
            behind[num] = edge_list[i] % bigmod;
            num ++;
        }
    }
    printf("The number of effective edges is %lld.\n", num);

    free(edge_list);

    unsigned long long *rows_addr;
    rows_addr = (unsigned long long *)malloc(sizeof(unsigned long long)*(dim+1));

    #pragma omp parallel for
    for (i = 0; i <= dim; i++)
        rows_addr[i] = bigmod;

    rows_addr[front[0]] = 0;
    #pragma omp parallel for
    for (i = 1; i < num; i++)
    {
        if (front[i] > front[i-1])
            rows_addr[front[i]] = i;
    }
    rows_addr[0] = 0;
    rows_addr[dim] = num;
    for (i = dim - 1; i > 0; i--)
    {
        if (rows_addr[i] == bigmod)
            rows_addr[i] = rows_addr[i+1];
    }
    // for (i = 0; i < dim; i++)
    //      printf("rid: %d, rwn: %d\n", i, rows_addr[i]);

    unsigned long long *revedge_list; 
    revedge_list = (unsigned long long *)malloc(sizeof(unsigned long long)*num);

    #pragma omp parallel for
    for (i = 0; i < num; i++)
        revedge_list[i] = behind[i] * bigmod + front[i];

    qsort(revedge_list, num, sizeof(unsigned long long), comp_ll);
    //printf("second sort is over!\n");

    unsigned long *col;
    unsigned long *row;
    col = (unsigned long *)malloc(sizeof(unsigned long)*num);
    row = (unsigned long *)malloc(sizeof(unsigned long)*num);

    #pragma omp parallel for
    for (i = 0; i < num; i++)
    {
        col[i] = revedge_list[i] / bigmod;
        row[i] = revedge_list[i] % bigmod;
    }

    free(revedge_list);
    //printf("column checked!\n");

    unsigned long long *cols_addr;
    cols_addr = (unsigned long long *)malloc(sizeof(unsigned long long)*(dim+1));

    #pragma omp parallel for
    for (i = 0; i <= dim; i++)
        cols_addr[i] = bigmod;

    cols_addr[col[0]] = 0;
    #pragma omp parallel for
    for (i = 1; i < num; i++)
    {
        if (col[i] > col[i-1])
            cols_addr[col[i]] = i;
    }
    cols_addr[0] = 0;
    cols_addr[dim] = num;
    for (i = dim - 1; i > 0; i--)
    {
        if (cols_addr[i] == bigmod)
            cols_addr[i] = cols_addr[i+1];
    }
    // for (i = 0; i < dim; i++)
    //      printf("cid: %d, cln: %d\n", i, cols_addr[i]);
    printf("Data preprocessing is completed!\n");

    unsigned long *val_mx;
    val_mx = (unsigned long *)malloc(sizeof(unsigned long)*num);
    //unsigned long rid, cid;
    //unsigned long long r_chs, c_chs, r_loc, c_loc, rl_n, cl_n;
    #pragma omp parallel for
    for (i = 0; i < num; i++)
    {
        unsigned long rid = front[i];
        unsigned long cid = behind[i];
        
        unsigned long long r_loc = rows_addr[rid];
        unsigned long long c_loc = cols_addr[cid];
        unsigned long long rl_n = rows_addr[rid+1];
        unsigned long long cl_n = cols_addr[cid+1];
        unsigned long long r_chs = rl_n - r_loc;
        unsigned long long c_chs = cl_n - c_loc;

        val_mx[i] = 0;

        while ((r_loc < rl_n) && (c_loc < cl_n))
        {
            if (behind[r_loc] == row[c_loc])
            {
                val_mx[i]++;
                r_loc++;
                c_loc++;
            }
            else if (behind[r_loc] > row[c_loc])
                c_loc++;
            else
                r_loc++;

        }

        if (i%dim == 1)
            printf("Counting is processing %6.3f%%\n", (i + 0.0)/num*100.0);
        
    }

    unsigned long long tri_ttl = 0;
    //#pragma omp parallel for reduction(+:tri_ttl)
    for (i = 0; i < num; i++)
        tri_ttl += val_mx[i];

    printf("Time costs %ld sec.\n", time(NULL) - t0 );
    printf("There are \033[1m%lld\033[0m triangles in the input graph.\n", tri_ttl);

    free(front);
    free(behind);
    free(col);
    free(row);
    free(rows_addr);
    free(cols_addr);
    free(val_mx);
    return 0;
}

