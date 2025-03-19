# Compiler
CXX = mpicxx

# Compiler flags
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -MMD -MP

# HDF5 include and library paths
HDF5_INC = /usr/local/hdf5/include
HDF5_LIB = /usr/local/hdf5/lib
HDF5_LIBS = -lhdf5 -lhdf5_cpp

# LAPACK include and library paths
# Replace the following with the correct path to your LAPACK installation
LAPACK_INC = /home/wuct/lapack-3.12.0/LAPACKE/include
LAPACK_LIB = -llapacke -llapack -lcblas -lrefblas -lgfortran -lblas

# FFTW library (add if used)
FFTW_LIBS = -lfftw3 -lm

# Output binary name
BIN_DIR = ../bin
TARGET = $(BIN_DIR)/solver

# Source file
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:%.cpp=$(BIN_DIR)/%.o)
DEPS = $(OBJS:.o=.d)

# Build target
all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -I$(HDF5_INC) -I$(LAPACK_INC) -L$(HDF5_LIB) $(OBJS) -o $(TARGET) $(HDF5_LIBS) $(FFTW_LIBS) $(LAPACK_LIB)
# $(CXX) $(CXXFLAGS) -I$(HDF5_INC) -I$(PETSC_INC) -L$(HDF5_LIB) -L$(PETSC_LIB) $(OBJS) -o $(TARGET) $(HDF5_LIBS) -lpetsc $(FFTW_LIBS)

$(BIN_DIR)/%.o: %.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(HDF5_INC) -I$(LAPACK_INC) -c $< -o $@
#	$(CXX) $(CXXFLAGS) -I$(HDF5_INC) -I$(PETSC_INC) -c $< -o $@

# Include dependency files if they exist
-include $(DEPS)

# Clean target
clean:
	rm -rf $(TARGET) $(OBJS)

.PHONY: all clean

