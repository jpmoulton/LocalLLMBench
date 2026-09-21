import { titleCase } from './title-case';

describe('titleCase', () => {
  test('capitalises each word', () => {
    expect(titleCase('one two')).toEqual('One Two');
  });
});
