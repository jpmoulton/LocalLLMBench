import { total } from './list-total';

describe('total', () => {
  test('empty array totals zero', () => {
    expect(total([])).toEqual(0);
  });

  test('sums the values', () => {
    expect(total([1, 2, 3])).toEqual(6);
  });
});
