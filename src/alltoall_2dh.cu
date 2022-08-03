/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include "strided_copy.h"

__thread int mem_stride_copy_gridsize = 1;
__thread int mem_stride_copy_blocksize = 1;
__thread int local_size = 1;
__thread void* scratch_buff = NULL;

void print_header() {
  PRINT("# %10s  %12s  %8s  %6s            out-of-place                       in-place          \n", "", "", "", "");
  PRINT("# %10s  %12s  %8s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %8s  %6s", size, count, typeName, opName);
}

void AlltoAll2DHGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = (count/nranks)*nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = count/nranks;
}

testResult_t AlltoAll2DHInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;
  // PRINT("\ninit data, nranks %d, sendcount %ld, recvcount %ld. sendBytes %ld, expectedBytes %ld \n", nranks, sendcount, recvcount, args->sendBytes, args->expectedBytes);

  CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, memStrideCopyUInt4Kernel));
  CUDACHECK(cudaGetDeviceCount(&local_size));

  for (int i=0; i<args->nGpus; i++) {
    char* str = getenv("NCCL_TESTS_DEVICE\n");
    int gpuid = str ? atoi(str) : args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    CUDACHECK(cudaSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, type, rep, rank));
    for (int j=0; j<nranks; j++) {
      TESTCHECK(InitData(((char*)args->expected[i])+args->sendBytes/nranks*j, sendcount/nranks, type, rep+rank*sendcount/nranks, j));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAll2DHGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  // PRINT("nranks %d, count %ld, typesize %d", nranks, count, typesize);
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAll2DHRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  CUDACHECK(cudaGetDeviceCount(&local_size));
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  int local_rank;
  NCCLCHECK(ncclCommCuDevice(comm, &local_rank));
  size_t rankOffset = count * wordSize(type);
  if (count == 0) return testSuccess;
  PRINT("RUN COLL send count %ld local_size %d\n", count, local_size);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#elif SCCL_SUPPORT
  NCCLCHECK(msccl2DAllToAll(sendbuff, recvbuff, count, type, comm, stream));
  return testSuccess;
#else
  if (nRanks % local_size != 0) {
    printf("AlltoAll: nranks %d is not a multiple of local_size %d\n", nRanks, local_size);
    return testNcclError;
  }
  int nnodes = nRanks / local_size;
  if (!(local_size == 1 || nnodes == 1)) {
    int node_rank = rank / local_size;
    PRINT("node rank %d\n", node_rank);

    size_t slice_size = count * wordSize(type) / nRanks;
    size_t slice_size_uint4 = slice_size / sizeof(uint4);

    // phase 0. per-gpu (ngpus) stride copy
    if (slice_size < sizeof(uint4)) {
      memStrideCopyCharKernel<<<mem_stride_copy_gridsize, mem_stride_copy_blocksize, 0, stream>>>(
        (char*)scratch_buff, (char*)sendbuff, slice_size, local_size, nnodes);
    } else {
      memStrideCopyUInt4Kernel<<<mem_stride_copy_gridsize, mem_stride_copy_blocksize, 0, stream>>>(
        (uint4*)scratch_buff, (uint4*)sendbuff, slice_size_uint4, local_size, nnodes);
    }

    // phase 1. intra-node alltoall
    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < local_size; g++) {
      NCCLCHECK(ncclSend(((char*)scratch_buff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * local_size, comm, stream));
      NCCLCHECK(ncclRecv(((char*)sendbuff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * local_size, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());

    // phase 2. per-gpu (nnodes) stride copy
    if (slice_size < sizeof(uint4)) {
      memStrideCopyCharKernel<<<mem_stride_copy_gridsize, mem_stride_copy_blocksize, 0, stream>>>(
        (char*)scratch_buff, (char*)sendbuff, slice_size, nnodes, local_size);
    } else {
      memStrideCopyUInt4Kernel<<<mem_stride_copy_gridsize, mem_stride_copy_blocksize, 0, stream>>>(
        (uint4*)scratch_buff, (uint4*)sendbuff, slice_size_uint4, nnodes, local_size);
    }

    // phase 3. inter-node alltoall
     NCCLCHECK(ncclGroupStart());
    for (int n = 0; n < nnodes; n++) {
      NCCLCHECK(ncclSend(((char*)scratch_buff) + n * local_size * slice_size, local_size * slice_size, ncclInt8, n * local_size + local_rank, comm, stream));
      NCCLCHECK(ncclRecv(((char*)sendbuff) + n * local_size * slice_size, local_size * slice_size, ncclInt8, n * local_size + local_rank, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    local_size = 1;
    return testSuccess;
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, type, r, comm, stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, type, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    local_size = 1;
    return testSuccess;
  }
#endif
}

struct testColl alltoAll2DHTest = {
  "AlltoAll2DH",
  AlltoAll2DHGetCollByteCount,
  AlltoAll2DHInitData,
  AlltoAll2DHGetBw,
  AlltoAll2DHRunColl
};

void AlltoAll2DHGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAll2DHGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AlltoAll2DHRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAll2DHTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  CUDACHECK(cudaMalloc(&scratch_buff, args->maxbytes));

  for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", -1));
  }

  CUDACHECK(cudaFree(scratch_buff));
  return testSuccess;
}

struct testEngine alltoAll2DHEngine = {
  AlltoAll2DHGetBuffSize,
  AlltoAll2DHRunTest
};

#pragma weak ncclTestEngine=alltoAll2DHEngine
