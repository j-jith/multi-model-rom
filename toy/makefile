PETSC_DIR=/usr/lib/petsc
SLEPC_DIR=/usr/lib/slepc

CFLAGS =
CPPFLAGS =
LIBFILES =
MAIN = main
MAIN_OBJ = main.o

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

all: ${MAIN}

${MAIN}: ${MAIN_OBJ} ${OBJFILES} chkopts
	-${CLINKER} -o ${MAIN} ${MAIN_OBJ} ${OBJFILES} ${PETSC_LIB} ${SLEPC_LIB}
	${RM} ${MAIN_OBJ} ${OBJFILES}
