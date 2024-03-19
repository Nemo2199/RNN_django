# RNN_django
Detection of malicious dga domains using GRU recurrent neural network.

**Creating_RNN_model.py** - teaches a GRU RNN model using and saves it. Accuracy is 98,25%.
Training process:

<table align="center">
  <tr>
    <td>
      <img width="400" height="290" src="https://github.com/Nemo2199/RNN_django/assets/81017100/60124e59-cfbf-48f6-9eb3-8965fbd80f2e">
    </td>
    <td>
      <img width="400" height="290" src="https://github.com/Nemo2199/RNN_django/assets/81017100/b189cecc-b8b6-4fd6-b160-200c67d1718a">
    </td>
  </tr>
</table>

**dga_domains_full.csv** - dataset for teaching. Contains legal and malicious domains.

**django_web_interface** - web app for using previously made model. Contains a field to enter desired domain and predict if it is legal or malicious:

<table align="center">
  <tr>
    <td>
      <img width="400" height="250" src="https://github.com/Nemo2199/RNN_django/assets/81017100/b1f93624-c6db-4d84-b5be-c01d80d35225">
    </td>
    <td>
      <img width="400" height="230" src="https://github.com/Nemo2199/RNN_django/assets/81017100/849bfac4-9bc7-4530-8912-53df809cb769">
    </td>
  </tr>
</table>

