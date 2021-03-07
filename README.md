# DeepFakes detection project

<h3> Datasets</h3>
The two datasets I have used have been <a href="https://ai.facebook.com/datasets/dfdc/">DFDC</a> and <a href="https://github.com/ondyari/FaceForensics">FaceForensics++</a>. For the data extraction process I have extensively used code from <a href="https://github.com/selimsef/dfdc_deepfake_challenge">selimsef</a> and <a href="https://github.com/Megatvini/DeepFaceForgeryDetection">Megatvini</a>.

In this case torch dataset's functions are inside the folder ```datasets```. In the file ```datasets/builder.py``` there is a function which can be called to create the datasets.
