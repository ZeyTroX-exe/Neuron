package main

import "fmt"

type Neuron struct {
    weights []float64
    bias    float64
    lr      float64
}

func NewNeuron(numInputs int, lr float64) *Neuron {
    weights := make([]float64, numInputs)
    for i := range weights {
        weights[i] = 0.1
    }
    return &Neuron{weights: weights, bias: 0.0, lr: lr}
}

func (n *Neuron) Predict(inputs []float64) float64 {
    sum := 0.0
    for i := range inputs {
        sum += n.weights[i] * inputs[i]
    }

    return sum + n.bias
}

func (n *Neuron) Train(inputs []float64, target float64) {
    pred := n.Predict(inputs)
    err := target - pred

    for i := range inputs {
        n.weights[i] += n.lr * err * inputs[i]
    }
    n.bias += n.lr * err
}

func main() {
    x1 := []float64{1, 2, 3}
    target := []float64{10, 15, 20}

    n := NewNeuron(1, 0.01)

    for epoch := 0; epoch < 5000; epoch++ {
        for i := range x1 {
            inputs := []float64{x1[i]}
            n.Train(inputs, target[i])
        }
    }

    fmt.Println(n.Predict([]float64{50}))
}