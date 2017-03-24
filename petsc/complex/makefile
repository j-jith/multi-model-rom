CFLAGS =
CPPFLAGS =
LIBFILES =
MMR = mmr
MMR_OBJ = mmr_main.o file_io.o pod.o reduce.o eigen.o misc.o

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

all: ${MMR}

${MMR}: ${MMR_OBJ} ${OBJFILES} globals.h chkopts
	-${CLINKER} -o ${MMR} ${MMR_OBJ} ${OBJFILES} ${PETSC_LIB} ${SLEPC_LIB}
	${RM} ${MMR_OBJ} ${OBJFILES}
