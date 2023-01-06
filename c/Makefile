CXX       := gcc
CXX_FLAGS := -g -pg -Wall -D_LARGEFILE_SOURCE=1 -D_FILE_OFFSET_BITS=64 -Wl,--stack,10485760

ifeq ($(OS),Windows_NT)
$(info ******* Windows *******)
else
$(info ******* Linux *******)
endif

ifeq ($(OS),Windows_NT)
$(info No extra define)
else
CXX_FLAGS += -D__LINUX__
endif

BIN       := bin
SRC       := src
INCLUDE   := inc

LIBRARIES   := 
EXECUTABLE  := main


all: $(BIN)/$(EXECUTABLE)

run: clean all
	clear
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.c *.c
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) $^ -o $@ $(LIBRARIES) -lm

clean:
	-rm $(BIN)/*