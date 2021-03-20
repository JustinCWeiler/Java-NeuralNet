TARGET = NeuralTest
OBJECTS = Matrix.class Vector.class NeuralNet.class
CP = .:lib/json-simple.jar

.PHONY: all run clean

all: $(OBJECTS) $(TARGET).class
run: all
	java -cp "$(CP)" $(TARGET)
clean:
	rm *.class

%.class: %.java
	javac -cp "$(CP)" $<
