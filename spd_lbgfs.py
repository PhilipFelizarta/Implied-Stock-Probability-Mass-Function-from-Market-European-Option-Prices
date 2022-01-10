import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import csv

print("TensorFlow version:", tf.__version__)

scale = 4677.03
s0 = 4677.03/scale
u_scale = 1200
min_strike = 0
max_strike = 10000
gamma = 1e-4


expiry = 'Mon Jan 10 2022'
K_call = []
u_call = []
K_put = []
u_put = []
with open('data/spx_quotedata.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 3:
            print(row)
        elif line_count > 3:
            if row[0] == expiry:
                if float(row[11]) >= min_strike and float(row[11]) <= max_strike:
                    K_call.append(float(row[11])/scale)
                    K_put.append(float(row[11])/scale)
                    u_call.append(float(row[4])/u_scale)
                    u_put.append(float(row[15])/u_scale)
        line_count += 1

print("K call: ", len(K_call))
print("call prices: ", len(u_call))
print("put prices: ", len(u_put))

n = 1000
r_max = 2.0
r_min = 0.0

plt.scatter(np.array(K_call)*scale, np.array(u_call)*u_scale)
plt.title("SPX Market Call Option Prices Jan 10 2022")
plt.xlabel("Strike")
plt.ylabel("Price per Contract")
plt.axvline(x=s0*scale, linestyle="--", label="$S_0$")
plt.legend()
plt.savefig("figures/MarketCallPrices.png")
plt.show()

plt.scatter(np.array(K_put)*scale, np.array(u_put)*u_scale)
plt.title("SPX Market Put Option Prices Jan 10 2022")
plt.xlabel("Strike")
plt.ylabel("Price per Contract")
plt.axvline(x=s0*scale, linestyle="--", label="$S_0$")
plt.legend()
plt.savefig("figures/MarketPutPrices.png")
plt.show()


#Create Learned PMF
distr_params = tf.Variable(tf.ones([n,]))
soft_distr = tf.nn.softmax(distr_params)
#geometrically spaced around the at-the-money 
right_distr = (r_max-s0)*np.geomspace(1e-7, s0, num=int(n/2), dtype=np.float32)
left_distr = np.geomspace(1e-7, s0+1e-7, num=int(n/2)+300-98-20-5, dtype=np.float32)
x_distr = tf.constant(np.sort(np.unique(np.concatenate((1-(left_distr-1e-7), right_distr+s0)))))
scaled_x = s0*x_distr

def full_loss_function(params):
    soft_distr = tf.nn.softmax(params)
    #soft_distr = soft_distr/tf.reduce_sum(soft_distr)
    
    right_distr = (r_max-s0)*np.geomspace(1e-7, s0, num=int(n/2), dtype=np.float32)
    left_distr = np.geomspace(1e-7, s0+1e-7, num=int(n/2)+300-98-20-5, dtype=np.float32)
    x_distr = tf.constant(np.sort(np.unique(np.concatenate((1-(left_distr-1e-7), right_distr+s0)))))
    scaled_x = x_distr
    log_coeff = 0.5
    
    loss_call = 0
    loss_put = 0
    i = 0
    for k in K_call:
        xk_term = tf.math.maximum(scaled_x - k, 0)
        u_k = tf.tensordot(tf.multiply(xk_term, soft_distr), scaled_x, 1)*scale/u_scale
        loss_call += log_coeff*tf.square(tf.math.log(u_k+1e-7) - tf.math.log(u_call[i]+1e-7)) + (1-log_coeff)*tf.square(u_k - u_call[i])
        #loss_call += (1/(u_call[i] + 1e-7))*tf.square((u_k - u_call[i]))
        i += 1

    loss_call = 0.5*(1/len(K_call))*loss_call 

    i = 0
    for k in K_put:
        xk_term = tf.math.maximum(k - scaled_x, 0)
        u_k = tf.tensordot(tf.multiply(xk_term, soft_distr), scaled_x, 1)*scale/u_scale
        loss_put += log_coeff*tf.square(tf.math.log(u_k+1e-7) - tf.math.log(u_put[i]+1e-7)) + (1.0-log_coeff)*tf.square(u_k - u_put[i])
        #loss_put += (1/(u_put[i] + 1e-7))*tf.square((u_k - u_put[i]))
        i += 1

    loss_put = 0.5*(1/len(K_put))*loss_put 
    contin_loss = 0
    for i in range(n-1):
        contin_loss += tf.square(params[i] - params[i+1])/tf.abs(scaled_x[i] - scaled_x[i+1])
    full_loss = loss_call + loss_put + gamma*contin_loss
    return full_loss

def full_loss_and_grad(p):
    return tfp.math.value_and_gradient(full_loss_function, p)

def full_model(params, log_coeff):
    soft_distr = tf.nn.softmax(params)
    #soft_distr = soft_distr/tf.reduce_sum(soft_distr)
    right_distr = (r_max-s0)*np.geomspace(1e-7, s0, num=int(n/2), dtype=np.float32)
    left_distr = np.geomspace(1e-7, s0+1e-7, num=int(n/2)+300-98-20-5, dtype=np.float32)
    x_distr = tf.constant(np.sort(np.unique(np.concatenate((1-(left_distr-1e-7), right_distr+s0)))))
    scaled_x = s0*x_distr
    model_call = []
    model_put = []
    
    loss_call = 0
    loss_put = 0
    i = 0
    for k in K_call:
        xk_term = tf.math.maximum(scaled_x - k, 0)
        u_k = tf.tensordot(tf.multiply(xk_term, soft_distr), scaled_x, 1)*scale/u_scale
        loss_call += log_coeff*tf.square(tf.math.log(u_k+1e-7) - tf.math.log(u_call[i]+1e-7)) + (1-log_coeff)*tf.square(u_k - u_call[i])
        #loss_call += (1/(u_scale*(u_call[i] + 1e-7)))*tf.square((u_k - u_call[i]))
        model_call.append(u_k)
        i += 1

    loss_call = 0.5*(1/len(K_call))*loss_call 
    

    i = 0
    for k in K_put:
        xk_term = tf.math.maximum(k - scaled_x, 0)
        u_k = tf.tensordot(tf.multiply(xk_term, soft_distr), scaled_x, 1)*scale/u_scale
        loss_put += log_coeff*tf.square(tf.math.log(u_k+1e-7) - tf.math.log(u_put[i]+1e-7)) + (1-log_coeff)*tf.square(u_k - u_put[i])
        #loss_put += (1/(u_scale*(u_put[i] + 1e-7)))*tf.square((u_k - u_put[i]))
        model_put.append(u_k)
        i += 1

    loss_put = 0.5*(1/len(K_put))*loss_put 
    contin_loss = 0
    for i in range(n-1):
        contin_loss += tf.square(params[i] - params[i+1])/tf.abs(scaled_x[i] - scaled_x[i+1])
    
    return model_call, model_put, loss_call, loss_put, contin_loss
    
start = tf.ones([n,])
optim_results = tfp.optimizer.lbfgs_minimize(full_loss_and_grad, tolerance=1e-8,initial_position=start)


# Check that the search converged
print("Did Converge?:", optim_results.converged.numpy())
# Print out the total number of function evaluations it took.
print ("Function evaluations: %d" % optim_results.num_objective_evaluations)
print("Full Loss: ", full_loss_function(optim_results.position).numpy())

plt.title("Model Stock Price Probability Mass Function (SPX)")
plt.scatter(x_distr*scale, tf.nn.softmax(optim_results.position).numpy())
plt.xlabel("Stock Price")
plt.ylabel("P($S_T$ = Stock Price)")
plt.xlim(0.75*scale, 1.25*scale)
plt.axvline(x=s0*scale, linestyle="--", label="$S_0$")
plt.legend()
plt.savefig("figures/PMF.png")
plt.show()


model_call, model_put, loss_call, loss_put, contin_loss = full_model(optim_results.position, 0.5)
print("Continuity Loss: ", gamma*contin_loss.numpy())
eval_u_call = []
eval_u_put = []
for i in range(len(K_call)):
    eval_u_call.append(model_call[i].numpy())

for i in range(len(K_put)):
    eval_u_put.append(model_put[i].numpy())

plt.scatter(K_call, np.array(eval_u_call)*u_scale)
plt.scatter(K_call, np.array(u_call)*u_scale)
plt.xlim(0.75, 1.25)
plt.ylim(-100, 1500)
plt.title("Model and Market Call Option Prices")
plt.xlabel("Moneyness")
plt.ylabel("Call Price")
plt.savefig("figures/ModelCall.png")
plt.show()
print("Call Loss: ", loss_call.numpy())

error = u_scale*(np.array(eval_u_call) - np.array(u_call))
plt.scatter(K_call, error)
plt.xlim(0.3, 1.25)
plt.title("Call Option Residuals")
plt.xlabel("Moneyness")
plt.ylabel("$u_{model} - u_{true}$")
plt.savefig("figures/CallResiduals.png")
plt.show()

plt.scatter(K_put, np.array(eval_u_put)*u_scale)
plt.scatter(K_put, np.array(u_put)*u_scale)
plt.xlim(0.75, 1.25)
plt.ylim(-100, 1500)
plt.title("Model and Market Put Option Prices")
plt.xlabel("Moneyness")
plt.ylabel("Put Price")
plt.savefig("figures/ModelPut.png")
plt.show()
print("Put Loss: ", loss_put.numpy())

error = u_scale*(np.array(eval_u_put) - np.array(u_put))
plt.scatter(K_put, error)
plt.xlim(0.75, 1.5)
plt.title("Put Option Residuals")
plt.xlabel("Moneyness")
plt.ylabel("$u_{model} - u_{true}$")
plt.savefig("figures/PutResiduals.png")
plt.show()

