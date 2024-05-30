<h1>Binary Classification of Breast Tumors</h1>
<h2>About</h2>
<p>Applying a homemade neural network towards classifying breast tumors as malignant or benign using mean radius and texture measurements. Various models, learning rates, and random seeds were tested. Results are shown below. The neural network used in this project is explored more by the repository, https://github.com/HenryChen4/Neural_Network.</p>
<h4>Settings</h4>
<ul>
  <li>Learning rate: 3</li>
  <li>Epochs: 100</li>
  <li>Seed for weight init: 1</li>
  <li>Seed for randomized data: 1</li>
  <li># of train: 500</li>
  <li># of test: 69</li>
</ul>
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
<img width="935" alt="Screen Shot 2023-07-24 at 8 10 04 AM" src="https://github.com/HenryChen4/Tumor_Classification/assets/71111859/253d92ee-fe86-4705-a4ee-51d19bfc9f6c">
<h4>Results</h4>
<p>Training loss: [0.24815771]
Train accuracy: 0.892
Test loss: [0.23887341]
Test accuracy: 0.8840579710144928</p>
<h3>with L2 regularization</h3>
<h4>Settings</h4>
<ul>
  <li>Same as above</li>
  <li>Regularization rate: 0.002</li>
</ul>
<img width="843" alt="Screen Shot 2023-07-24 at 8 28 45 AM" src="https://github.com/HenryChen4/Tumor_Classification/assets/71111859/e63eea75-5f44-4563-ad48-d08c15613828">
<h4>Results</h4>
<p>Training loss: [0.24728433]
Train accuracy: 0.892
Test loss: [0.23611153]
Test accuracy: 0.8840579710144928</p>
<p>A very slight improvement in test loss. 0.2388 -> 0.2361</p>

