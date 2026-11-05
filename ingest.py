
import html
import os
import requests
import re


OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"


def reconstruct_abstract(inverted_index):
    """Convert OpenAlex inverted abstract index into readable text."""

    if not inverted_index:
        return "Abstract not available."

    words = {}

    for word, positions in inverted_index.items():
        for position in positions:
            words[position] = word

    return " ".join(
        words[position] for position in sorted(words)
    )


def search_research_papers(topic, number_of_papers=5, start_year=None):
    """Search OpenAlex first, then Crossref if OpenAlex is unavailable."""

    params = {
        "search": topic,
        "per-page": min(number_of_papers, 100),
        "sort": "relevance_score:desc"
    }

    if start_year:
        params["filter"] = f"from_publication_date:{start_year}-01-01"

    openalex_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if openalex_key:
        params["api_key"] = openalex_key

    openalex_error = None

    try:
        response = requests.get(
            OPENALEX_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            papers = []
            for item in results:
                authors = [
                    author.get("author", {}).get("display_name")
                    for author in item.get("authorships", [])
                    if author.get("author", {}).get("display_name")
                ]

                papers.append({
                    "title": item.get("title") or "Title unavailable",
                    "authors": authors,
                    "year": item.get("publication_year"),
                    "doi": item.get("doi"),
                    "venue": (
                        (item.get("primary_location", {}).get("source") or {})
                        .get("display_name", "Venue unavailable")
                    ),
                    "url": item.get("id", ""),
                    "abstract": reconstruct_abstract(
                        item.get("abstract_inverted_index")
                    )
                })
            return papers

    except (requests.exceptions.RequestException, ValueError) as error:
        openalex_error = error

    crossref_params = {
        "query": topic,
        "rows": min(number_of_papers, 100),
    }
    if start_year:
        crossref_params["filter"] = f"from-pub-date:{start_year}-01-01"

    try:
        response = requests.get(
            CROSSREF_URL,
            params=crossref_params,
            headers={
                "User-Agent": "ResearchNexusAI/1.0 (academic research assistant)"
            },
            timeout=30
        )
        response.raise_for_status()
        results = response.json().get("message", {}).get("items", [])
    except (requests.exceptions.RequestException, ValueError) as error:
        if openalex_error:
            raise RuntimeError(
                "Could not retrieve papers from OpenAlex or Crossref. "
                f"OpenAlex error: {openalex_error}. Crossref error: {error}"
            ) from error
        raise RuntimeError(
            f"Crossref could not retrieve papers: {error}"
        ) from error

    papers = []
    for item in results:
        authors = [
            " ".join(
                part for part in (
                    author.get("given", ""),
                    author.get("family", "")
                ) if part
            )
            for author in item.get("author", [])
        ]
        date_parts = (
            item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or item.get("created", {}).get("date-parts")
            or []
        )
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        doi = item.get("DOI")
        abstract = item.get("abstract") or "Abstract not available."
        abstract = html.unescape(re.sub(r"<[^>]+>", " ", abstract))
        abstract = re.sub(r"\s+", " ", abstract).strip()

        papers.append({
            "title": (item.get("title") or ["Title unavailable"])[0],
            "authors": authors,
            "year": year,
            "doi": f"https://doi.org/{doi}" if doi else None,
            "venue": (item.get("container-title") or ["Venue unavailable"])[0],
            "url": item.get("URL") or (
                f"https://doi.org/{doi}" if doi else ""
            ),
            "abstract": abstract
        })

    return papers


def save_papers(papers, filename="research_papers.txt"):
    """Save discovered paper metadata into a text file."""

    with open(filename, "w", encoding="utf-8") as file:

        for index, paper in enumerate(papers, start=1):

            file.write(f"\nPAPER {index}\n")
            file.write("=" * 60 + "\n")

            file.write(f"Title: {paper['title']}\n")
            file.write(f"Authors: {', '.join(paper['authors'])}\n")
            file.write(f"Year: {paper['year']}\n")
            file.write(f"DOI: {paper['doi'] or 'Not available'}\n")
            file.write(f"Venue: {paper['venue']}\n")
            file.write(f"Source: {paper['url']}\n")
            file.write(f"Abstract: {paper['abstract']}\n")

            file.write("\n")


if __name__ == "__main__":

    topic = input("Enter research topic: ")

    number = int(input("Number of papers to retrieve: "))

    year_input = input(
        "Starting publication year (press Enter to skip): "
    ).strip()

    start_year = int(year_input) if year_input else None

    papers = search_research_papers(
        topic,
        number,
        start_year
    )

    if papers:
        save_papers(papers)

        print(f"\nSuccessfully retrieved {len(papers)} papers.")

        for paper in papers:
            print("\nTitle:", paper["title"])
            print("Year:", paper["year"])
            print("DOI:", paper["doi"])

        print("\nPaper information saved in research_papers.txt")

    else:
        print("No research papers retrieved.")
