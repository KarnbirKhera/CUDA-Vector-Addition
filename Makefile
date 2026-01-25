NVCC = nvcc
FLAGS = -lineinfo
TEST_FLAGS = -lgtest -Xcompiler -pthread

#=============================================================================
#                              BASIC COMMANDS
#=============================================================================

build:
	$(NVCC) -o vectorSum vectorSum.cu main.cu $(FLAGS)

test:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	./test_vectorSum

benchmark:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	./test_vectorSum --gtest_filter=*Benchmark*:*ExportCSV*

correctness:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	./test_vectorSum --gtest_filter=Correctness.*

clean:
	rm -f vectorSum test_vectorSum *.ncu-rep *.nsys-rep *.sqlite

#=============================================================================
#                    PROFILING (requires NVIDIA Nsight tools)
#=============================================================================

# Profile all kernels once with NVIDIA Nsight Compute
# Install: https://developer.nvidia.com/nsight-compute
profile:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	ncu --set full --import-source yes --force-overwrite -o profile_report ./test_vectorSum --gtest_filter=Profile.SingleRun
	ncu-ui profile_report.ncu-rep &

# Timeline view with NVIDIA Nsight Systems
# Install: https://developer.nvidia.com/nsight-systems
timeline:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	nsys profile --stats=true --force-overwrite true -o timeline_report ./test_vectorSum --gtest_filter=Profile.SingleRun
	nsys-ui timeline_report.nsys-rep &

#=============================================================================
#                              HELP
#=============================================================================

help:
	@echo "CUDA Vector Addition Benchmark"
	@echo ""
	@echo "Basic commands:"
	@echo "  make build       - Compile the main executable"
	@echo "  make test        - Run all tests (benchmarks + correctness)"
	@echo "  make benchmark   - Run only benchmark tests"
	@echo "  make correctness - Run only correctness tests"
	@echo "  make clean       - Remove build artifacts"
	@echo ""
	@echo "Profiling (requires NVIDIA Nsight tools):"
	@echo "  make profile     - Profile with Nsight Compute"
	@echo "  make timeline    - Profile with Nsight Systems"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit"
	@echo "  - Google Test (libgtest-dev)"
	@echo "  - NVIDIA GPU"