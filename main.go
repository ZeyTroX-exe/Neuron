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

type Network struct {
	layers []Layer
	out    *Neuron
}

func NewNetwork(layers int, size int, lr float64) *Network {
	out := *NewNeuron(lr)
	lrs := make([]Layer, layers)

	for l := range lrs {
		lrs[l] = *NewLayer(size, lr)
	}

	return &Network{layers: lrs, out: &out}
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

func (net *Network) Train(input []float64, target float64) {
	input = scale(input)

	forward := []float64{}
	for n := range net.layers[0].neurons {
		forward = append(forward, math.Max(0, net.layers[0].neurons[n].Predict(input)))
	}

	guess := net.out.Predict(forward)
	for n := range net.layers[0].neurons {
		if forward[n] <= 0 {
			continue
		}
		err := (target - guess) * net.out.weights[n]
		net.layers[0].neurons[n].Adjust(err, input)
	}

	err := target - guess
	net.out.Adjust(err, forward)
}

func (net *Network) Predict(input []float64) float64 {
	input = scale(input)

	forward := []float64{}
	for n := range net.layers[0].neurons {
		forward = append(forward, math.Max(0, net.layers[0].neurons[n].Predict(input)))
	}
	guess := net.out.Predict(forward)
	return guess
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
		n.weights[w] += n.lr * err * input[w]
	}
	n.bias += n.lr * err
}

func scale(in []float64) []float64 {
	out := []float64{}
	for i := range in {
		out = append(out, in[i]/100)
	}
	return out
}

func main() {
	input := []float64{23, 51, 36, 61}
	input2 := []float64{13, 11, 32, 22}
	output := float64(31)
	output2 := float64(82)

	net := NewNetwork(1, 10, 0.001)

	for range 500 {
		net.Train(input, output)
		net.Train(input2, output2)
	}

	fmt.Println(net.Predict(input))

}

// todo: multi layer
