export const colormapDict: {[key: number]: string } = {
    0: 'jet',
    1: 'hsv',
    2: 'hot',
    3: 'cool',
    4: 'spring',
    5: 'summer',
    6: 'autumn',
    7: 'winter',
    8: 'bone',
    9: 'cooper',
    10: 'greys',
    11: 'rainbow',
    12: 'rainbow_soft',
    13: 'white',
    14: 'black',
    15: 'red',
    16: 'green',
    17: 'blue',
    18: 'yellow',
    19: 'cyan',
    20: 'fuchsia',
    21: 'terrain',
    22: 'ocean',
};
 
export const getIdByColormapName = (dictionary: { [key: number]: string }, colormapName: string): number | undefined => {
    for (const key in dictionary) {
      if (dictionary.hasOwnProperty(key)) {
        const value = dictionary[key];
        if (value === colormapName) {
          return parseInt(key); // Convert the key back to a number
        }
      }
    }
    return undefined; // Return undefined if the value is not found
}
