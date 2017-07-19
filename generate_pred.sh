# Some variables
currentDirectory=$(pwd)
mainDirectory="Documents/time_series/time-series-stocks"
condaEnvName="py35"

echo "Entering the main directory: $mainDirectory"
cd $mainDirectory

echo "Activating Conda enviroment: $condaEnvName"
source activate $condaEnvName

# The beefy part
echo "Training, generating and saving the predictions..."
python main.py

# TODO: update the web page
echo "TODO: convert predictions to JSON and update the page."

echo "Deactivating Conda enviroment..."
source deactivate

echo "Returning to the original directory..."
cd $currentDirectory
