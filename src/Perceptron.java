import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Perceptron {
    private double[] weights;// each weight corresponds to an input feature.
    private double threshold; // if the weighted sum of inputs will exceed threshold, the perceptron will fire
    // sigma (weight * feature) >= threshold -> 1, otherwise 0
    private double learningRate;

    public Perceptron(int inputCount, double learningRate) {
        this.weights = new double[inputCount];
        this.threshold = Math.random() * 0.1;
        this.learningRate = learningRate;

        // here we init the weights with small random values
        for (int i = 0; i < inputCount; i++) {
            weights[i] = Math.random() * 0.1; // then we will adjust based on the errors
        }
    }

    public int predict(double[] inputs) {
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum >= threshold ?  1 : 0;
    }

    public void train(double[] inputs, int expectedOutput) {
        int prediction = predict(inputs);
        int error = expectedOutput - prediction; // -1 0 1

        // update weights and threshold based on the error
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * inputs[i];
        }
        threshold -= learningRate * error;
        // 1 - perceptron under-predicts (output 0 instead of 1), means error is positive.
        // The update to thresh. decreases it and makes it easier for the perceptron to output 1
        // in the future since the weighted sum of inputs doesn't need to be as large to exceed the threshold.
        // 2 - perceptron over-predicts (1 instead of 0), means error is neg.
        // The update the thresh will increase it and makes it harder for the perceptron to output 1
        // since the weighted sum now needs to be larger to exceed the new, higher threshold.
    }

    public static List<double[]> readData(String filePath) throws IOException {
        List<double[]> data = new ArrayList<>();
        Map<String, Integer> labelToNumeric = new HashMap<>();
        labelToNumeric.put("Iris-virginica", 1);
        labelToNumeric.put("Iris-versicolor", 0);

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] observations = new double[parts.length];
                for (int i = 0; i < parts.length - 1; i++) {
                    observations[i] = Double.parseDouble(parts[i]);
                }
                observations[parts.length - 1] = labelToNumeric.get(parts[parts.length - 1]);
                data.add(observations);
            }
        }
        return data;
    }

    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter the path to the training data file:");
        String trainingPath = scanner.nextLine();

        System.out.println("Enter the path to the test data file:");
        String testPath = scanner.nextLine();

        System.out.println("Enter the learning rate:");
        double learningRate = Double.parseDouble(scanner.nextLine());

        System.out.println("Enter the number of epochs:");
        int epochs = Integer.parseInt(scanner.nextLine());

        List<double[]> trainingData = readData(trainingPath);
        List<double[]> testData = readData(testPath);

        // init perceptron
        Perceptron perceptron = new Perceptron(trainingData.get(0).length - 1, learningRate);

        // training
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(trainingData);
            for (double[] data : trainingData) {
                double[] inputs = Arrays.copyOfRange(data, 0, data.length - 1);
                int expectedOutput = (int) data[data.length - 1];
                perceptron.train(inputs, expectedOutput);
            }

            // after each epoch we test the accuracy
            int correctCount = 0;
            for (double[] data : testData) {
                double[] inputs = Arrays.copyOfRange(data, 0, data.length - 1);
                int expectedOutput = (int) data[data.length - 1];
                if (perceptron.predict(inputs) == expectedOutput) {
                    correctCount++;
                }
            }
            System.out.println("Epoch " + (epoch + 1) + " Accuracy: " + (double) correctCount / testData.size());
        }

        while (true) {
            System.out.println("Enter input features separated by commas or type 'exit':");
            String inputLine = scanner.nextLine();
            if ("exit".equalsIgnoreCase(inputLine)) break;

            double[] inputs = Arrays.stream(inputLine.split(","))
                    .mapToDouble(Double::parseDouble)
                    .toArray();
            int prediction = perceptron.predict(inputs);
            System.out.println("Predicted class: " + prediction);
        }
        scanner.close();
    }
}
