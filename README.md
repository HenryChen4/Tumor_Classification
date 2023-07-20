<h1>Binary Classification of Breast Tumors</h1>
<h2>About</h2>
<p>Applying a neural network towards classifying breast tumors as malignant or benign using mean radius and texture measurements. Various models, learning rates, and random seeds were tested. Results are shown below. The neural network used in this project is explored more by the repository, https://github.com/HenryChen4/Neural_Network.</p>
<h2>Tests</h2> 
<h3>Test 1, prior to regularization</h3>
<h4>Settings</h4>
<ul>
  <li>Learning rate: 1.5</li>
  <li>Epochs: 200</li>
  <li>Seed for weight init: 5</li>
  <li>Seed for randomized data: 1</li>
  <li># of train: 500</li>
  <li># of test: 69</li>
</ul>
<h4>Model Structure</h4>
<table class="tg">
<thead>
  <tr>
    <th class="tg-sg5v">Layer 1</th>
    <th class="tg-0pky">Layer 2</th>
    <th class="tg-0pky">Layer 3</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Units: 3<br>Activation: Relu</td>
    <td class="tg-0pky">Units: 2<br>Activation: Relu</td>
    <td class="tg-0pky">Units: 1<br>Activation: Sigmoid</td>
  </tr>
</tbody>
</table>
<img width="893" alt="Screen Shot 2023-07-20 at 12 04 34 AM" src="https://github.com/HenryChen4/Tumor_Classification/assets/71111859/6ec7ac46-c57f-41eb-bf19-f8395191cc47">
<h4>Results</h4>
Training error: 0.10361329
Test error: 0.10095054
Train accuracy: 0.898
Test accuracy: 0.8985507246376812

