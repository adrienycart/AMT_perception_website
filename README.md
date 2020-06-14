# Automatic Music Transcription listening tests website

This repository holds the code for the website used to run the listening tests described in:

Adrien Ycart, Lele Liu, Emmanouil Benetos and Marcus Pearce, 2020. "Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription", _Transactions of the International Society for Music Information Retrieval (TISMIR)_, 3(1), pp.68–81.

```  
    @article{ycart2019PEAMT,
       Author = {Ycart, Adrien and Liu, Lele and Benetos, Emmanouil and Pearce, Marcus},    
       Booktitle = {Transactions of the International Society for Music Information Retrieval (TISMIR)},    
       Title = {Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription},       
       Year = {2020},
       Volume = {3},
       Issue = {1},
       Pages = {68--81},
       DOI = {http://doi.org/10.5334/tismir.57},
    }  




## Requirements

This project uses Python 2.7.15. The website is designed using Flask. The database is managed using SQLAlchemy. The list of required dependencies is available in `requirements.txt`
To install the list of dependencies, run:
`pip install -r requirements.txt`

## Data

All the audio examples should be placed in: `./app/static`
They should be in a folder containing one subfolder per example, and inside this subfolder, one mp3 per system + a reference mp3 named `target.mp3`

You can set the relative path of the parent of all the examples inside the `./app/static` folder by modifying the variable `DATA_PATH` in `./config.py`

## Running the website

Once the environment is set up, create the database with: `flask db init`

Then, start the website with: `flask run`

You can navigate the website at [http://localhost:5000](http://localhost:5000)

## More info

This website's design was heavily inspired by Miguel Grinberg's [Flask Mega Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world).
This is a very useful resource to understand Flask in general, and in particular how this specific website was designed.
