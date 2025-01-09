import unittest

from preparedata.loaddata.XmlProcessor import XmlProcessor
from preparedata.loaddata.tests.MockDataLoader import MockDataLoader

# Mock column mappings
COLUMN_MAPPINGS = {
    ('shoton', 'team', 'home'),
    ('shoton', 'team', 'away'),
    ('possession', 'homepos', 'home'),
    ('possession', 'awaypos', 'away')
}


class TestXmlProcessor(unittest.TestCase):

    def setUp(self):
        self.data = MockDataLoader().load_data()
        self.xml_processor = XmlProcessor(self.data)

    def test_parse_shoton(self):
        result = self.xml_processor.process_data()

        self.assertTrue(len(result) > 0)
        self.assertTrue(result.head(1)['away_shoton'][0] == 5)
        self.assertFalse('shoton' in result.columns)

    def test_parse_possession(self):
        result = self.xml_processor.process_data()

        self.assertTrue(len(result) > 0)
        self.assertTrue(result.head(1)['away_possession'][0] == 42)
        self.assertFalse('possession' in result.columns)


if __name__ == '__main__':
    unittest.main()
