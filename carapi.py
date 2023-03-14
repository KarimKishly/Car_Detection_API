import requests
from prettytable import PrettyTable


def findcar(car_name):
    entry = car_name

    url = f'https://public.opendatasoft.com/api/records/1.0/search/?dataset=all-vehicles-model&q={entry.lower()}&sort=modifiedon&facet=make&facet=model&facet=cylinders&facet=drive&facet=eng_dscr&facet=fueltype&facet=fueltype1&facet=mpgdata&facet=phevblended&facet=trany&facet=vclass&facet=year'

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data)
        records = data['records']
        table = PrettyTable(['Make', 'Model', 'Year', 'Drive', 'Engine displacement', 'Engine Description', 'Fuel Type 1'])
        if len(records) > 0:
            for record in records:
                fields = record['fields']
                table.add_row(
                    [fields['make'], fields['model'], fields['year'], fields['drive'], str(fields['displ']) + ' L', fields['trany'], fields['fueltype1']])
            print(table)

            # add more fields as needed
        else:
            print('No records found for', entry)
    else:
        print('Request failed with status code', response.status_code)
