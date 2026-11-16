package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Neuron struct {
	weights []float64
	bias    float64
	lr      float64
}

type Layer struct {
	neurons []Neuron
}

func NewNeuron(lr float64) *Neuron {
	return &Neuron{lr: lr, bias: 0.2}
}

func NewLayer(size int, lr float64) *Layer {
	l := make([]Neuron, size)

	for i := range l {
		l[i] = *NewNeuron(lr)
	}

	return &Layer{neurons: l}
}

func (n *Neuron) Predict(input []float64) float64 {
	if n.weights == nil {
		n.weights = make([]float64, len(input))
		for i := range len(input) {
			n.weights[i] = rand.Float64()
		}
	}

	out := n.bias
	for i := range input {
		out += input[i] * n.weights[i]
	}

	return out
}

func (n *Neuron) Adjust(err float64, input []float64) {
	for w := range n.weights {
		n.weights[w] += err * input[w] * n.lr
	}

	n.bias += err * n.lr
}

func main() {
	input := []float64{23, 51, 36, 61}
	output := float64(31)

	out := NewNeuron(0.1)
	hidden := NewLayer(5, 0.1)

	for range 60 {
		forward := []float64{}
		for n := range hidden.neurons {
			prediction := hidden.neurons[n].Predict(input)
			forward = append(forward, math.Max(0, prediction))
		}

		guess := out.Predict(forward)
		fmt.Println(guess)

		for n := range hidden.neurons {
			if forward[n] == 0 {
				continue
			}
			err := (output - guess) * out.weights[n]

			hidden.neurons[n].Adjust(err, input)
		}

		err := output - guess

		out.Adjust(err, forward)
	}
}

// todo:Network struct
