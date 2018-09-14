# Bayesian

I've changed the weight_loss to :

	def weight_loss(self,cur_grad,pre_grad):
		res = []
		for i in range(len(pre_grad)):
			res.append(K.sum(K.square(cur_grad[i] - pre_grad[i])))
		return sum(res)/len(res)
    
The confusion matrix is ![confusion matrix](https://github.com/mozzielol/Bayesian/blob/master/confusion%20matrix.png)
