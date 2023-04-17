import requests
import os
import random


def get_recordings(query, page=1):
    url = "https://xeno-canto.org/api/2/recordings"
    params = {"query": query, "page": page}
    response = requests.get(url, params=params)
    return response.json()


def save_audio_file(url, folder, file_name):
    response = requests.get(url)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, file_name), "wb") as f:
        f.write(response.content)


def get_random_species(species_list, num_species, exclude_species):
    random_species = []
    while len(random_species) < num_species:
        species = random.choice(species_list)
        if species not in exclude_species:
            random_species.append(species)
            exclude_species.append(species)
    return random_species


def get_all_species(recordings):
    species_list = []
    for recording in recordings:
        species = f"{recording['gen']} {recording['sp']}"
        if species not in species_list:
            species_list.append(species)
    return species_list


bird_species = [
    "troglodytes troglodytes",  # Eurasian Wren
    "turdus merula",  # Common Blackbird
    "parus major",  # Great Tit
    "erithacus rubecula",  # European Robin
    "alauda arvensis",  # Eurasian Skylark
]

other_species_count = 10
recordings_per_species = 10

# Fetch the complete list of species
recordings_result = get_recordings("cnt:poland")
all_species = get_all_species(recordings_result["recordings"])

# Choose 10 random bird species, excluding those already fetched
random_species = get_random_species(all_species, other_species_count, bird_species)

for species in random_species:
    print(f"Fetching recordings for {species}...")
    page = 1
    species_folder = f"other/{species.replace(' ', '_')}"
    recordings_count = 0

    while recordings_count < recordings_per_species:
        result = get_recordings(species, page)
        if not result["recordings"]:
            break

        for recording in result["recordings"]:
            if recordings_count >= recordings_per_species:
                break

            file_url = recording["file"]
            file_name = recording["file-name"]
            save_audio_file(file_url, species_folder, file_name)
            recordings_count += 1

        page += 1

    print(f"{recordings_count} recordings for {species} have been fetched and saved.")

print("Other species' recordings fetched and saved successfully.")
