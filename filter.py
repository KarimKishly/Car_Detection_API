def filterClass(line):
    remove_list = ['Sedan', 'Type-S', 'Type R', 'Coupe', 'SUV', 'Convertible', 'Crew Cab', 'Minivan', 'Hatchback', 'Ron Fellows Edition', 'EXT', '-Class', 'Passenger Van', 'Superleggera', '570-4', 'IPL', 'Extended Cab', 'Regular Cab', 'Supercab', 'Super Duty', 'Wagon', 'Club Cab', 'Quad Cab', '3500', 'Extended Cab', 'Regular Cab', 'Classic Extended Cab', 'SUT']
    for word in remove_list:
        line = line.replace(word, '')
    return line
s = 'C\\Users\\ferrari.jpeg'
print(s.replace('.', '_cropped.').split('\\')[-1])