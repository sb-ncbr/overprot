
#installation folder for MAPSCI
BIN      = ../bin

#source code folder for MAPSCI
SRC      = ../src

CC       = g++

CXXFLAGS  = -O3 -DNDEBUG -s -static -fomit-frame-pointer -felide-constructors \
          -falign-functions=4 -funroll-loops -finline-functions \
          -fexpensive-optimizations -falign-jumps=2 \
          -frerun-loop-opt -floop-optimize

CPPFLAGS  = -I$(SRC) -I$(SRC)/align -I$(SRC)/geom -I$(SRC)/pdbutils -I$(SRC)/utils -I$(SRC)/alglib

VPATH   = $(SRC) $(SRC)/align $(SRC)/geom $(SRC)/pdbutils $(SRC)/utils $(SRC)/alglib

SOURCES = alignment.cpp \
	angles.cpp \
	atomiter.cpp \
	clique.cpp \
	consensus.cpp \
	global.cpp \
	mapsci.cpp \
	msvd.cpp \
	multiple.cpp \
	pairwise.cpp \
	params.cpp \
	pdbatom.cpp \
	pdbfilter.cpp \
	point.cpp \
	protein.cpp \
	runs.cpp \
	semiglobal.cpp \
	summary.cpp \
	utils.cpp \
	zscore.cpp \
	\
	ap.cpp \
	bdsvd.cpp \
	bidiagonal.cpp \
	blas.cpp \
	lq.cpp \
	qr.cpp \
	reflections.cpp \
	rotations.cpp \
	svd.cpp


mapsci: alignment.o angles.o atomiter.o clique.o consensus.o global.o msvd.o \
	multiple.o pairwise.o params.o pdbatom.o pdbfilter.o \
	point.o protein.o runs.o semiglobal.o summary.o utils.o zscore.o \
	ap.o bdsvd.o blas.o bidiagonal.o lq.o qr.o \
	reflections.o rotations.o svd.o


include $(subst .cpp,.d,$(SOURCES))


%.d: %.cpp

	$(CC) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

install:
	mv mapsci $(BIN) 