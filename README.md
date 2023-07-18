<h1>Binary Classification of Breast Tumors</h1>
<h2>About</h2>
<p>Applying a neural network towards classifying breast tumors as malignant or benign using mean radius and texture measurements. Various models, learning rates, and random seeds were tested. Results are shown below. The neural network used in this project is explored more by the repository, https://github.com/HenryChen4/Neural_Network.</p>
<h3>1st run, prior to regularization</h3>
<p>
    Model structure:
    <br>
    3 units, 2 units, 1 unit
    <br>
    Relu, Relu, Sigmoid
    <br>
    <br>
    Parameters:
    <br>
    100 sets of testing data
    <br>
    460 sets of training data
    <br>
    learning rate: 1.5
    <br>
    epochs: 200
    <br>
    seed: 5
    <br>

    Final training error: 0.0839
    <br>
    Final testing error: 0.3835
    <br>
    <br>

    A clear example of overfitting. A side note, overfitting was attempted to be reduced through increasing training data, normalizing training data, and adjusting seed. Results were insufficient. 
    <br>
</p>
