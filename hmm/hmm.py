import os
import pickle
import numpy as np
import scipy as sp
dlname='all_obs.obj'

if not os.path.exists('./%s'%dlname):
	fpaths = []
	labels = []
	spoken = []
	for f in os.listdir('audio'):
	    for w in os.listdir('audio/' + f):
		fpaths.append('audio/' + f + '/' + w)
		labels.append(f)
		if f not in spoken:
		    spoken.append(f)
	print 'Words spoken:',spoken


	
	from scipy.io import wavfile
	import numpy as np

	data = np.zeros((len(fpaths), 2000000))
	maxsize = -1
	for n,file in enumerate(fpaths):
	    _, d = wavfile.read(file)
	    if d.ndim>1:
		d=np.mean(d,axis=1);
	    data[n, :d.shape[0]] = d
	    if d.shape[0] > maxsize:
		maxsize = d.shape[0]
	data = data[:, :maxsize]

	print 'Number of files total:',data.shape[0]
	all_labels = np.zeros(data.shape[0])
	for n, l in enumerate(set(labels)):
	    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
	    
	print 'Labels and label indices',all_labels




	import scipy
	import numpy as np

	def stft(x, fftsize=64, overlap_pct=.5):   
	    hop = int(fftsize * (1 - overlap_pct))
	    w = scipy.hanning(fftsize + 1)[:-1]    
	    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
	    return raw[:, :(fftsize / 2)]


	import matplotlib.pyplot as plt
	plt.plot(data[0, :], color='steelblue')
	plt.title('Timeseries example for %s'%labels[0])
	plt.xlim(0, 3500)
	plt.xlabel('Time (samples)')
	plt.ylabel('Amplitude (signed 16 bit)')
	plt.figure()

	log_freq = 20 * np.log(np.abs(stft(data[0, :])))
	print log_freq.shape
	plt.imshow(log_freq, cmap='gray', interpolation=None)
	plt.xlabel('Freq (bin)')
	plt.ylabel('Time (overlapped frames)')
	plt.ylim(log_freq.shape[1])
	plt.title('PSD of %s example'%labels[0])



	from numpy.lib.stride_tricks import as_strided

	def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
	    win_size = l_size + r_size + c_size
	    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
	    strides = x.strides + (x.strides[-1],)
	    xs = as_strided(x, shape=shape, strides=strides)
	    def is_peak(x):
		centered = (np.argmax(x) == l_size + int(c_size/2))
		l = x[:l_size]
		c = x[l_size:l_size + c_size]
		r = x[-r_size:]
		passes = np.max(c) > np.max([f(l), f(r)])
		if centered and passes:
		    return np.max(c)
		else:
		    return -1
	    r = np.apply_along_axis(is_peak, 1, xs)
	    top = np.argsort(r, None)[::-1]
	    heights = r[top[:n_peaks]]
	    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
	    return heights, top[:n_peaks]




	from numpy.lib.stride_tricks import as_strided

	def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
	    win_size = l_size + r_size + c_size
	    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
	    strides = x.strides + (x.strides[-1],)
	    xs = as_strided(x, shape=shape, strides=strides)
	    def is_peak(x):
		centered = (np.argmax(x) == l_size + int(c_size/2))
		l = x[:l_size]
		c = x[l_size:l_size + c_size]
		r = x[-r_size:]
		passes = np.max(c) > np.max([f(l), f(r)])
		if centered and passes:
		    return np.max(c)
		else:
		    return -1
	    r = np.apply_along_axis(is_peak, 1, xs)
	    top = np.argsort(r, None)[::-1]
	    heights = r[top[:n_peaks]]
	    #Add l_size and half - 1 of center size to get to actual peak location
	    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
	    return heights, top[:n_peaks]


	
	all_obs = []
	for i in range(data.shape[0]):
	    d = np.abs(stft(data[i, :]))
	    n_dim = 6
	    obs = np.zeros((n_dim, d.shape[0]))
	    for r in range(d.shape[0]):
		_, t = peakfind(d[r, :], n_peaks=n_dim)
		obs[:, r] = t.copy()
	    if i % 10 == 0:
		print "Processed obs %s"%i
	    all_obs.append(obs)
	    
	all_obs = np.atleast_3d(all_obs)


	 

	fob= open('all_obs.obj', 'w') 
	pickle.dump(all_obs,fob);


else:
    print '%s already created!'%dlname
    file_pi2 = open('all_obs.obj', 'r') 
    all_obs= pickle.load(file_pi2)




def pdf_multivariate_gauss(x,mu,cov):
    '''
    Caculate the multivariate normal density (pdf)
    
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    '''
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))
    
    
    
class gmmhmm:
    
    def __init__(self, n_states):
        self.n_states = n_states
        self.random_state = np.random.RandomState(0)
        
        
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))
        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))
        
        self.mu = None
        self.covs = None
        self.n_dims = None
           
    def _forward(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
         
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha
    
    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape);
           
        beta[:, -1] = np.ones(B.shape[0])
            
        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return beta
    
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states):
            
            B[s, :] = pdf_multivariate_gauss(obs.T, self.mu[:, s].T, self.covs[:, :, s].T)
            
        return B
    
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
    
    def _em_init(self, obs):
        #Using this _em_init function allows for less required constructor args
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]
        return self
    
    def _em_step(self, obs): 
        obs = np.atleast_2d(obs)
        B = self._state_likelihood(obs)
        T = obs.shape[1]
        
        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)
        
        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, T))
        
        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi_sum += self._normalize(partial_sum)
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = self._normalize(partial_g)
              
        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = self._normalize(partial_g)
        
        expected_prior = gamma[:, 0]
        expected_A = self._stochasticize(xi_sum)
        
        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
        
        gamma_state_sum = np.sum(gamma, axis=1)
        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
            #Symmetrize
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
        
        #Ensure positive semidefinite by adding diagonal loading
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]
        
        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        return log_likelihood
    
    def fit(self, obs, n_iter=15):
        
        if len(obs.shape) == 2:
            for i in range(n_iter):
                self._em_init(obs)
                log_likelihood = self._em_step(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(n_iter):
                    self._em_init(obs[n, :, :])
                    log_likelihood = self._em_step(obs[n, :, :])
        return self
    
    def transform(self, obs):
        
        if len(obs.shape) == 2:
            B = self._state_likelihood(obs)
            log_likelihood, _ = self._forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self._state_likelihood(obs[n, :, :])
                log_likelihood, _ = self._forward(B)
                out[n] = log_likelihood
            return out

if __name__ == "__main__":
    rstate = np.random.RandomState(0)
    t1 = np.ones((4, 40)) + .001 * rstate.rand(4, 40)
    t1 /= t1.sum(axis=0)
    t2 = rstate.rand(*t1.shape)
    t2 /= t2.sum(axis=0)
    
    m1 = gmmhmm(2)
    m1.fit(t1)
    m2 = gmmhmm(2)
    m2.fit(t2)
    
    m1t1 = m1.transform(t1)
    m2t1 = m2.transform(t1)
    print "Likelihoods for test set 1"
    print "M1:",m1t1
    print "M2:",m2t1
    print "Prediction for test set 1"
    print "Model", np.argmax([m1t1, m2t1]) + 1
    print 
    
    m1t2 = m1.transform(t2)
    m2t2 = m2.transform(t2)
    print "Likelihoods for test set 2"
    print "M1:",m1t2
    print "M2:",m2t2
    print "Prediction for test set 2"
    print "Model", np.argmax([m1t2, m2t2]) + 1



from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(all_labels, test_size=0.1, random_state=0)

for n,i in enumerate(all_obs):
    all_obs[n] /= all_obs[n].sum(axis=0)

for train_index, test_index in sss:
    X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
    y_train, y_test = all_labels[train_index], all_labels[test_index]
print 'Size of training matrix:', X_train.shape
print 'Size of testing matrix:', X_test.shape



ys = set(all_labels)
ms = [gmmhmm(6) for y in ys]
_ = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
ps = [m.transform(X_test) for m in ms]
res = np.vstack(ps)
predicted_labels = np.argmax(res, axis=0)
missed = (predicted_labels != y_test)
print 'Test accuracy:%.2f percent'%(100 * (1 - np.mean(missed)))



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_labels)
plt.matshow(cm, cmap='gray')
ax = plt.gca()
_ = ax.set_xticklabels([" "] + [l[:2] for l in spoken])
_ = ax.set_yticklabels([" "] + spoken)
plt.title('Confusion matrix, single speaker')
plt.ylabel('True label')
plt.xlabel('Predicted label')
