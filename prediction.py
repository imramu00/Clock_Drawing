from detecto import utils, visualize, core
import torch

def predict(img):
	model = torch.load("torch_model_pkl.pkl")
	predictions = model.predict_top(img)
	labels, boxes, scores = predictions

	l = labels
	s = []
	for i in scores:
		s.append(i.item())
	result = {"labels":l, "scores":s}

	return result
