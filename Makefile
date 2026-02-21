NVCC = nvcc
FLAGS = -lineinfo -arch=sm_89 -O3
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

# Define metrics as a variable
L2_METRICS = l2_tex_read_hit_rate,l2_tex_write_hit_rate,l2_tex_hit_rate,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_hit.sum,lts__t_sectors_op_write_hit.sum

# Detailed L2 analysis with sector counts
l2-detailed:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	ncu --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sector_hit_rate.pct ./test_vectorSum --gtest_filter=Profile.SingleRun

l2-full:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	ncu --metrics lts__t_sector_hit_rate,lts__t_sector_op_read_hit_rate,lts__t_sector_op_write_hit_rate,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum ./test_vectorSum --gtest_filter=Profile.NaiveL2FullTest

l2-validate:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	ncu --set full --import-source yes --force-overwrite -o l2_validate_report ./test_vectorSum --gtest_filter=Profile.L2WritePolicyValidation
	ncu-ui l2_validate_report.ncu-rep &

ILPTest:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	ncu --set full --import-source yes --force-overwrite -o profile_report ./test_vectorSum --gtest_filter=Profile.ILPTest
	ncu-ui profile_report.ncu-rep &

profile-fp8:
	$(NVCC) -o test_vectorSum vectorSum.cu vectorSumTest.cu $(FLAGS) $(TEST_FLAGS)
	ncu --set full --import-source yes --force-overwrite -o fp8_comparison_report \
	./test_vectorSum --gtest_filter=Profile.FP32_vs_FP8
	ncu-ui fp8_comparison_report.ncu-rep &
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
	@echo "  make l2-hits     - Check L2 read vs write hit rates"
	@echo "  make l2-detailed - Detailed L2 sector analysis"
	@echo "  make profile     - Profile with Nsight Compute (full)"
	@echo "  make timeline    - Profile with Nsight Systems"