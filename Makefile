CC = g++
WEIGHTTYPE = double
CFLAGS = -D WEIGHTTYPE=$(WEIGHTTYPE) -lm -O2 -funroll-loops -ffast-math -Irnnlm-0.4b -std=c++11
#CFLAGS = -lm -O2 -Wall

all: rnnlmlib.o rnnlm

rnnlmlib.o : rnnlm-0.4b/rnnlmlib.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) -c rnnlm-0.4b/rnnlmlib.cpp -o rnnlm-0.4b/rnnlmlib.o

rnnlm : rnnlmMod.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) rnnlmMod.cpp rnnlm-0.4b/rnnlmlib.o -o rnnlmMod

clean:
	rm -rf *.o rnnlm
