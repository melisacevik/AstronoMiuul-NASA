def compare_crater_area_with_countries(crater_area):
    """
    Krater yüzey alanını, verilen ülkelerin yüzey alanları ile karşılaştırır.
    """
    country_areas = {
        'Russia': 17098242,  # km^2
        'Canada': 9976140,
        'China': 9596961,
        'United States': 9525067,
        'Brazil': 8515767,
        'Australia': 7692024,
        'India': 3287263,
        'Argentina': 2780400,
        'Kazakhstan': 2724900,
        'Algeria': 2381741,
        'Democratic Republic of the Congo': 2344858,
        'Greenland': 2166086,
        'Saudi Arabia': 2149690,
        'Mexico': 1964375,
        'Indonesia': 1904569,
        'Sudan': 1861484,
        'Libya': 1759540,
        'Iran': 1648195,
        'Mongolia': 1564110,
        'Peru': 1285216,
        'Chad': 1284000,
        'Niger': 1267000,
        'Angola': 1246700,
        'Mali': 1240192,
        'South Africa': 1221037,
        'Colombia': 1141748,
        'Ethiopia': 1104300,
        'Bolivia': 1098581,
        'Mauritania': 1030700,
        'Egypt': 1002450,
        'Tanzania': 945087,
        'Nigeria': 923768,
        'Venezuela': 916445,
        'Pakistan': 881913,
        'Namibia': 825615,
        'Mozambique': 801590,
        'Turkey': 783562,
        'Chile': 756102,
        'kiyaslanmayacak': 17098243
    }

    # Krater yüzey alanını en yakın ülkenin yüzey alanı ile karşılaştırma
    closest_country = min(country_areas, key=lambda x: abs(country_areas[x] - crater_area))

    if closest_country == 'kiyaslanmayacak':
        return "Kraterin yüzey alanı, kıyaslanmayacak ölçüde büyüktür."
    else:
        return f"Kraterin yüzey alanı, {closest_country}'nin yüzey alanına ({country_areas[closest_country]} km^2) yakındır."

