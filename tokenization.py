""" Utilities to extract tokens from a text """
import re
import decimal
import datetime


class Token:
    """ Represents an interesting text entity.

    Attributes
    ----------
    source: Parser
        Reference to the parser that has generated the token.

    line_number: int
        Line number in the text where the token occurred.

    start_position: int
        Position on the line where the token starts.

    end_position: int
        Position on the line where the token ends.

    type: string
        Indicates the type of token. Currently implemented are 'Numeric' and
        'Date'

    value: type-dependent
        The value of the token represented as an appropriate type.
        'decimal.Decimal' for 'Numeric' tokens
        'datetime.date' for 'Date' tokens
    """
    def __str__(self):
        return str((self.line_number, self.start_position, self.end_position,
                    self.type, self.value))

    @property
    def source_line(self):
        """ Return the line of the text containing the token. """
        return self.source.lines[self.line_number]

    @property
    def previous_line(self):
        """ Return the line of the text preceeding the token. """
        if self.line_number == 0:
            return ''
        else:
            return self.source.lines[self.line_number - 1]

    @property
    def line_prefix(self):
        """ Return the part of the source line preceeding the token. """
        return self.source_line[:self.start_position]

    @property
    def string(self):
        """ Returns the token as it occurred in the source text. """
        return self.source_line[self.start_position:self.end_position]


class Parser:
    def __init__(self, text):
        """ Parses a given text into tokens.

        Due to the unstructured nature of the texts that we process, the
        tokenization has some differences compared to a typical parser.

        Firstly, not every character in the source text is assigned to a
        token.

        Second, the extracted tokens are not guaranteed to be disjoint, so
        a single character in the text may be assigned to more than one
        token. The disambiguation is left for the later processing stages
        as there is not enough lexical information to handle all possible
        cases in the parser.
        """
        self.lines = text.splitlines()
        self.tokens = []
        for self.line_number, self.current_line in enumerate(self.lines):
            self._extract_numbers()
            self._extract_dates()

    def _extract_numbers(self):
        # match numbers with and witout thousands separators
        # i.e. 1234.56 as well as 1,234.56
        # allow dots, commas and spaces as either the separators
        # or the decimal sign
        regex = '((?:\d+|\d\d?\d?(?:[., ]\d\d\d)+)[., ]\d\d)(?:\s|$)'
        #regex = '(\d+\.\d\d)(?:\s|$)'
        for match in re.finditer(regex, self.current_line):
            start_position, end_position = match.span()
            tmp = re.sub('\D', '', match.group())
            value = decimal.Decimal(tmp[:-2] + '.' + tmp[-2:])
            self._add_token('Numeric', start_position, end_position, value)

    def _extract_dates(self):
        # match dates of the form day/month/year or 'day-month-year'
        # where appropriate interpret as month/day/year as well
        regex = '(\d?\d)[-/](\d?\d)[-/]((?:\d\d)?\d\d)'
        for match in re.finditer(regex, self.current_line):
            start_position, end_position = match.span()
            day, month, year = [int(x) for x in match.groups()]

            try:
                value = datetime.date(*self._adjust_date(year, month, day))
                self._add_token('Date', start_position, end_position, value)
            except ValueError as e:
                pass

            if month <= 12 and day <= 12:
                try:
                    value = datetime.date(*self._adjust_date(year, day, month))
                    self._add_token('Date', start_position, end_position, value)
                except ValueError as e:
                    pass

        # match dates of the form 'year-month-day'
        regex = '((?:\d\d)?\d\d)-(\d?\d)-(\d?\d)'
        for match in re.finditer(regex, self.current_line):
            start_position, end_position = match.span()
            year, month, day = [int(x) for x in match.groups()]

            try:
                value = datetime.date(*self._adjust_date(year, month, day))
                self._add_token('Date', start_position, end_position, value)
            except ValueError as e:
                pass

        month_values = {'january': 1, 'jan': 1, 'jnr': 1, 'jny': 1,
                        'february': 2, 'feb': 2, 'fbr': 2, 'fby': 2,
                        'march': 3, 'mar': 3,
                        'april': 4, 'apr': 4,
                        'may': 5,
                        'june': 6, 'jun': 6,
                        'july': 6, 'jul': 7,
                        'august': 8, 'aug': 8,
                        'september': 9, 'sep': 9,
                        'october': 10, 'oct': 10,
                        'november': 11, 'nov': 11,
                        'december': 12, 'dec': 12}
        month_re = '|'.join(month_values.keys())

        # match dates with spelled out months of the form 'day month, year'
        regex = '(\d?\d)(?:st|nd|th)?[- \t]?({})[,.]?[- \t]?((?:\d\d)?\d\d)'.\
                format(month_re)
        for match in re.finditer(regex, self.current_line, re.IGNORECASE):
            start_position, end_position = match.span()
            day, month, year = match.groups()
            year, day = int(year), int(day)
            month = month_values[month.lower()]

            try:
                value = datetime.date(*self._adjust_date(year, month, day))
                self._add_token('Date', start_position, end_position, value)
            except ValueError as e:
                pass

        # match dates with spelled out months of the form 'month day, year'
        regex = '({})[ \t](\d?\d)[.,]?[ \t]((?:\d\d)?\d\d)'.format(month_re)
        for match in re.finditer(regex, self.current_line, re.IGNORECASE):
            start_position, end_position = match.span()
            month, day, year = match.groups()
            year, day = int(year), int(day)
            month = month_values[month.lower()]

            try:
                value = datetime.date(*self._adjust_date(year, month, day))
                self._add_token('Date', start_position, end_position, value)
            except ValueError as e:
                pass

    def _adjust_date(self, year, month, day):
        if year < 100:
            year = 2000 + year
        if month > 12:
            day, month = month, day

        return (year, month, day)

    def _add_token(self, token_type, start_position, end_position, value):
        token = Token()
        token.source = self
        token.line_number = self.line_number
        token.start_position = start_position
        token.end_position = end_position
        token.type = token_type
        token.value = value
        self.tokens.append(token)

    def contain_word(self, string):
        result = any(string.upper() in re.findall(r"\w+", line.upper())
            for line in self.lines)
        return result
#        return any((.find(string.upper()) > -1)
#                    for line in self.lines)

    def contain_word_exact(self, string):
        return any((line.upper().find(string.upper()) > -1)
            for line in self.lines)

    @property
    def numeric_tokens(self):
        return [token for token in self.tokens if token.type == 'Numeric']

    @property
    def date_tokens(self):
        return [token for token in self.tokens if token.type == 'Date']
