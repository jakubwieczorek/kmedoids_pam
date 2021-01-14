# KMedoids PAM algorithm

## About
This repository conatins KMedoids PAM algorithm implementation in Python version 3.8 for numerical and nominal values.

## Get started
Prepare the data and save it for example in the cars3.csv file with semicolon as delimiter:
|        |Price [PLN]|Max speed [km/h]|Acceleration to 100 km/h [s]|
|--------------|-----------|----------------|----------------------------|
|Nissan NV200  |50890      |165             |15                          |
|Nissan Micra  |61900      |178             |11.8                        |
|Nissan Juke   |76930      |180             |10.7                        |
|Nissan Qashqai|98480      |198             |9.9                         |
|Nissan X-Trail|115440     |198             |11.5                        |
|Nissan Leaf   |123900     |157             |7.3                         |
|Nissan Navara |147000     |184             |10.8                        |
|Nissan GT-R   | 527000    |315             |2.9                         |
|BMW Seria 1   |106400     |250             |4.8                         |
|BMW Seria 2   |113700     |250             |4.6                         |
|BMW X1        |133900     |235             |6.5                         |
|BMW X2        |139100     |250             |5                           |
|BMW Seria 4   |172900     |250             |4.5                         |
|BMW Seria 5   |197900     |305             |3.4                         |
|BMW Seria 6   |260900     |250             |5.3                         |
|Dacia Dokker  |41550      |173             |12.5                        |
|Dacia Duster  |42900      |200             |10.4                        |
|Dacia Lodgy   |61800      |185             |10.9                        |
|Dacia Sandero |32900      |182             |11.5                        |


Run the script with 3 clusters specified in the argument:
```
python3 KMedoids.py -i cars3.csv -c 3 --delimiter=";"
```
Result is displayed on the plot and stored in the json file cars3_result.json, with medoids as Nissan NV200 and BMW X1 and Nissan GT-R:
```json
{
    "Nissan NV200": [
        "Nissan NV200",
        "Nissan Micra",
        "Nissan Juke",
        "Dacia Dokker",
        "Dacia Duster",
        "Dacia Lodgy",
        "Dacia Sandero"
    ],
    "BMW X1": [
        "Nissan Qashqai",
        "Nissan X-Trail",
        "Nissan Leaf",
        "Nissan Navara",
        "BMW Seria 1",
        "BMW Seria 2",
        "BMW X1",
        "BMW X2",
        "BMW Seria 4",
        "BMW Seria 5",
        "BMW Seria 6"
    ],
    "Nissan GT-R": [
        "Nissan GT-R"
    ]
}
```
## Questions or need help?
Don't hesitate to send me a mail on jakub.wieczorek0101@gmail.com.

## Copyright and license
kmedoids_pam project is copyright to Jakub Wieczorek under the [MIT License](https://opensource.org/licenses/MIT).
