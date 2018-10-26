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

// 快速索引判断函数
int comp_ul(const void *a, const void *b)
{
    if (*(unsigned long *)a < *(unsigned long *)b)
        return -1;
    else
        return 1;
}

void par_sort(long long n, unsigned long dim, unsigned long *a, unsigned long *b, unsigned long long *a_loc)
{
    long i;
    unsigned long *a_count;
    a_count = (unsigned long *)malloc(sizeof(unsigned long)*dim);

    #pragma omp parallel for
    for (i = 0; i < dim; i++)
        a_count[i] = 0;
    
    for (i = 0; i < n; i++)
        a_count[a[i]]++;

    a_loc[0] = 0;
    for (i = 1; i <= dim; i++)
        a_loc[i] = a_loc[i-1] + a_count[i-1];
    
    free(a_count);

    unsigned long *b_count;
    b_count = (unsigned long *)malloc(sizeof(unsigned long)*dim);
    #pragma omp parallel for
    for (i = 0; i < dim; i++)
        b_count[i] = 0;

    unsigned long *b_sort;
    b_sort = (unsigned long *)malloc(sizeof(unsigned long)*n);
    long long a_loc_new;
    for (i = 0; i < n; i++)
    {
        a_loc_new = a_loc[a[i]] + b_count[a[i]];
        b_sort[a_loc_new] = b[i];
        b_count[a[i]]++;
    }
    
    free(b_count);

    #pragma omp parallel for
    for (i = 0; i < dim; i++)
    {
        long loc_0 = a_loc[i];
        long loc_n = a_loc[i+1];
        long ln = loc_n - loc_0;
        if (ln > 0)
        {
            unsigned long *b_sec;
            b_sec = (unsigned long *)malloc(sizeof(unsigned long)*(ln));
            long k;
            for (k = 0; k < ln; k++)
            {
                b_sec[k] = b_sort[loc_0+k];
                a[loc_0+k] = i;
            }
            
            qsort(b_sec, ln, sizeof(unsigned long), comp_ul);
            for (k = 0; k < ln; k++)
                b[loc_0+k] = b_sec[k];
            free(b_sec);
        }
    }

    free(b_sort);
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
    unsigned long *front;
    unsigned long *behind;
    front = (unsigned long *)malloc(sizeof(unsigned long)*n_edge);
    behind = (unsigned long *)malloc(sizeof(unsigned long)*n_edge);

    unsigned long dim = 0;

    //#pragma omp parallel for
    for (i = 0; i < n_edge; i++)
    {
        if (endpoints[2*i] < endpoints[2*i+1])
        {
            front[i] = endpoints[2*i];
            behind[i] = endpoints[2*i+1];
            if (endpoints[2*i+1] > dim)
                dim = endpoints[2*i+1];
        }
        else
        {
            front[i] = endpoints[2*i+1];
            behind[i] = endpoints[2*i];
            if (endpoints[2*i] > dim)
                dim = endpoints[2*i];
        }
    }
    dim++;
    printf("The quantity of nodes is %ld.\n", dim);

    munmap(endpoints, dt_size);
    close(fp);

    printf("Data preprocessing starts!\n");
    unsigned long long *rows_addr;
    rows_addr = (unsigned long long *)malloc(sizeof(unsigned long long)*(dim+1));

    par_sort(n_edge, dim, front, behind, rows_addr);

    short *vld;
    vld = (short *)malloc(sizeof(short)*n_edge);
    if (front[0] == behind[0])
        vld[0] = 0;
    else
        vld[0] = 1;
        
    #pragma omp parallel for
    for (i = 1; i < n_edge; i++)
    {
        if (front[i] == behind[i])
            vld[i] = 0;
        else
        {
            if (front[i] == front[i-1])
            {
                if (behind[i] == behind[i-1])
                    vld[i] = 0;
                else
                    vld[i] = 1;
            }
            else
                vld[i] = 1;
        }
    }

    unsigned long long num = 0;
    unsigned long long *vidx;
    vidx = (unsigned long long *)malloc(sizeof(unsigned long long)*n_edge);
    for (i = 0; i < n_edge; i++)
    {
        num += vld[i];
        if (vld[i] > 0)
            vidx[i] = num-1;
        else
            vidx[i] = 0;
    }
        
    printf("The number of effective edges is %lld.\n", num);

    unsigned long *row_0;
    unsigned long *col_0;
    unsigned long *col;
    unsigned long *row;
    row_0 = (unsigned long *)malloc(sizeof(unsigned long)*num);
    col_0 = (unsigned long *)malloc(sizeof(unsigned long)*num);
    col = (unsigned long *)malloc(sizeof(unsigned long)*num);
    row = (unsigned long *)malloc(sizeof(unsigned long)*num);
    //printf("Sub 1.0\n");
    #pragma omp parallel for
    for (i = 0; i < n_edge; i++)
    {
        if (vld[i] > 0)
        {
            long long idx = vidx[i];
            row_0[idx] = front[i];
            col_0[idx] = behind[i];
            row[idx] = front[i];
            col[idx] = behind[i];
        }
    }

    free(front);
    free(behind);

    unsigned long *r_count;
    r_count = (unsigned long *)malloc(sizeof(unsigned long)*dim);
    #pragma omp parallel for
    for (i = 0; i < dim; i++)
        r_count[i] = 0;
    
    for (i = 0; i < num; i++)
        r_count[row_0[i]]++;

    rows_addr[0] = 0;
    for (i = 1; i <= dim; i++)
        rows_addr[i] = rows_addr[i-1] + r_count[i-1];
    free(r_count);

    unsigned long long *cols_addr;
    cols_addr = (unsigned long long *)malloc((sizeof(unsigned long long))*(dim+1));

    par_sort(num, dim, col, row, cols_addr);
    
    printf("Data preprocessing is completed!\n");

    unsigned long *val_mx;
    val_mx = (unsigned long *)malloc(sizeof(unsigned long)*num);

    #pragma omp parallel for
    for (i = 0; i < num; i++)
    {
        unsigned long rid = row_0[i];
        unsigned long cid = col_0[i];
        
        unsigned long long r_loc = rows_addr[rid];
        unsigned long long c_loc = cols_addr[cid];
        unsigned long long rl_n = rows_addr[rid+1];
        unsigned long long cl_n = cols_addr[cid+1];
        unsigned long long r_chs = rl_n - r_loc;
        unsigned long long c_chs = cl_n - c_loc;

        val_mx[i] = 0;
        while ((r_loc < rl_n) && (c_loc < cl_n))
        {
            if (col_0[r_loc] == row[c_loc])
            {
                val_mx[i]++;
                r_loc++;
                c_loc++;
            }
            else if (col_0[r_loc] > row[c_loc])
                c_loc++;
            else
                r_loc++;
        }

        if (i%dim == 1)
            printf("Counting is processing %6.2f%%\n", (i + 0.0)/num*100.0);
        
    }

    unsigned long long tri_ttl = 0;
    #pragma omp parallel for reduction(+:tri_ttl)
    for (i = 0; i < num; i++)
        tri_ttl += val_mx[i];

    printf("Time costs %ld sec.\n", time(NULL) - t0 );
    printf("There are \033[1m%lld\033[0m triangles in the input graph.\n", tri_ttl);

    free(col_0);
    free(row_0);
    free(col);
    free(row);
    free(rows_addr);
    free(cols_addr);
    free(val_mx);
    return 0;
}

