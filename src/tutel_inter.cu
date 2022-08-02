
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

void print_header() {
  PRINT("# %10s  %12s  %8s  %6s            out-of-place                       in-place          \n", "", "", "", "");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type", "redop",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %8s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

void print_line_header (size_t size, size_t count, const char *typeName, size_t wordSize, const char *opName, int root) {
  PRINT("%12li  %12li  %8s  %6s", size, count, typeName, opName);
}

testResult_t TutelInterStageGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = (count/nranks)*nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = count/nranks;
  return testSuccess;
}

testResult_t TutelInterStageInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, memStrideCopyUInt4Kernel));
  CUDACHECK(cudaGetDeviceCount(&local_size));

  for (int i=0; i<args->nGpus; i++) {
    char* str = getenv("NCCL_TESTS_DEVICE");
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

void TutelInterStageGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t TutelInterStageRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  CUDACHECK(cudaGetDeviceCount(&local_size));
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  int local_rank = rank % local_size;
  // NCCLCHECK(ncclCommCuDevice(comm, &local_rank));
  size_t rankOffset = count * wordSize(type);
  if (count == 0) return testSuccess;

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#elif SCCL_SUPPORT
  NCCLCHECK(mscclTutelInterStage(sendbuff, recvbuff, count, type, comm, stream));
  return testSuccess;
#else
  if (nRanks % local_size != 0) {
    printf("TutelInterStage: nranks %d is not a multiple of local_size %d\n", nRanks, local_size);
    return testNcclError;
  }
  int nnodes = nRanks / local_size;
  if (!(local_size == 1 || nnodes == 1)) {
    size_t slice_size = count * wordSize(type) / nRanks;

    // only phase 3. inter-node alltoall
    NCCLCHECK(ncclGroupStart());
    for (int n = 0; n < nnodes; n++) {
      NCCLCHECK(ncclSend(((char*)sendbuff) + n * local_size * slice_size, local_size * slice_size, ncclInt8, n * local_size + local_rank, comm, stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff) + n * local_size * slice_size, local_size * slice_size, ncclInt8, n * local_size + local_rank, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, type, r, comm, stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, type, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
  }
#endif
}

struct testColl TutelInterStageTest = {
  "InterStage",
  TutelInterStageGetCollByteCount,
  TutelInterStageInitData,
  TutelInterStageGetBw,
  TutelInterStageRunColl
};

testResult_t TutelInterStageRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &TutelInterStageTest;
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
  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", -1));
  }

  return testSuccess;
}

testResult_t TutelInterStageGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks, int rank) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  TESTCHECK(TutelInterStageGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks, rank));
  return testSuccess;
}

struct testEngine TutelInterStageEngine = {
  TutelInterStageGetBuffSize,
  TutelInterStageRunTest
};


#pragma weak ncclTestEngine=TutelInterStageEngine