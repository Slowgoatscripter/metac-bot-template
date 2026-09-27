import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    _is_real_env,
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class SummerTemplateBot2026(ForecastBot):
    """
    This is the template bot for Summer 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Summer AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    # Forecasting models to take turns with, one per forecast. Set after
    # construction; empty means every forecast uses llms["default"]. With two
    # models and an even number of forecasts per question, each question gets
    # an equal share from each model, and the median blends them.
    forecasters: list[GeneralLlm] = []

    def _next_forecaster(self, question: MetaculusQuestion) -> GeneralLlm:
        if not self.forecasters:
            return self.get_llm("default", "llm")
        # Count per question: questions are forecast concurrently, so a single
        # shared counter could hand one question an uneven mix.
        turns = self.__dict__.setdefault("_forecaster_turns", {})
        turn = turns.get(question.page_url, 0)
        turns[question.page_url] = turn + 1
        llm = self.forecasters[turn % len(self.forecasters)]
        logger.info(f"Forecast {turn + 1} for {question.page_url} uses {llm.model}")
        return llm

    # Web-search model that looks up base rates before the news is read. Set
    # after construction; None skips that step.
    base_rate_researcher: GeneralLlm | None = None
    # Also pull news articles from AskNews, on top of the web search.
    use_asknews: bool = False

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            news, articles, base_rates = await asyncio.gather(
                self._research_news(question),
                self._research_asknews(question),
                self._research_base_rates(question),
            )
            sections = [
                ("Base rates (how often things like this happen)", base_rates),
                ("Current news about this question", news),
                ("Recent news articles (AskNews)", articles),
            ]
            research = "\n\n".join(
                f"## {title}\n{text}" for title, text in sections if text
            )
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _research_asknews(self, question: MetaculusQuestion) -> str:
        if not self.use_asknews:
            return ""
        try:
            return await AskNewsSearcher().get_formatted_news_async(
                question.question_text
            )
        except Exception as e:
            # A second news source helps but isn't essential.
            logger.warning(f"AskNews research failed for {question.page_url}: {e}")
            return ""

    async def _research_base_rates(self, question: MetaculusQuestion) -> str:
        if self.base_rate_researcher is None:
            return ""
        resolves = (
            question.scheduled_resolution_time.strftime("%Y-%m-%d")
            if question.scheduled_resolution_time
            else "unknown"
        )
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster. Your job is the outside view: before anyone looks at the specifics of this question, how often do events like this one happen?

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}

            Today is {datetime.now().strftime("%Y-%m-%d")}. The question is scheduled to resolve on {resolves}.

            1. Name one to three reference classes: groups of past situations comparable to this one (for example "US midterm elections since 1946" or "monthly US CPI releases over the last 10 years").
            2. For each, search for the historical record and give the base rate as a number with its source. Match the time window to this question: if it resolves in 3 months, say how often the outcome happens within a 3-month window. For numeric or date questions, give the typical values and how much they usually move over a period that long.
            3. Say which reference class fits best and why, and note anything that makes this case unusual.

            Do not forecast this question and do not summarize current news; another assistant covers the news. If no sensible reference class exists, say so in one sentence.
            """
        )
        try:
            return await self.base_rate_researcher.invoke(prompt)
        except Exception as e:
            # Base rates help but aren't essential; forecast on the news alone.
            logger.warning(f"Base-rate research failed for {question.page_url}: {e}")
            return ""

    async def _research_news(self, question: MetaculusQuestion) -> str:
        research = ""
        researcher = self.get_llm("researcher")

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )

        if isinstance(researcher, GeneralLlm):
            research = await researcher.invoke(prompt)
        elif (
            researcher == "asknews/news-summaries"
            or researcher == "asknews/deep-research/low-depth"
            or researcher == "asknews/deep-research/medium-depth"
            or researcher == "asknews/deep-research/high-depth"
        ):
            research = await AskNewsSearcher().call_preconfigured_version(
                researcher, prompt
            )
        elif researcher.startswith("smart-searcher"):
            model_name = researcher.removeprefix("smart-searcher/")
            searcher = SmartSearcher(
                model=model_name,
                temperature=0,
                num_searches_to_run=2,
                num_sites_per_search=10,
                use_advanced_filters=False,
            )
            research = await searcher.invoke(prompt)
        elif not researcher or researcher == "None" or researcher == "no_research":
            research = ""
        else:
            research = await self.get_llm("researcher", "llm").invoke(prompt)
        return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The base rate: the reference class from your research assistant that fits this question best, and how often the outcome happens in it over a similar length of time.
            (b) The time left until the outcome to the question is known.
            (c) The status quo outcome if nothing changed.
            (d) A brief description of a scenario that results in a No outcome.
            (e) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters start from the base rate and adjust it for what is specific to this case, and put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._next_forecaster(question).invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The base rate: the reference class from your research assistant that fits this question best, and how often each kind of outcome happens in it.
            (b) The time left until the outcome to the question is known.
            (c) The status quo outcome if nothing changed.
            (d) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters start from the base rate and adjust it for what is specific to this case, (2) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (3) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        reasoning = await self._next_forecaster(question).invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The base rate: what the historical record from your research assistant says about typical outcomes and how much they usually move over a similar length of time.
            (b) The time left until the outcome to the question is known.
            (c) The outcome if nothing changed.
            (d) The outcome if the current trend continued.
            (e) The expectations of experts and markets.
            (f) A brief description of an unexpected scenario that results in a low outcome.
            (g) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters start from the base rate and adjust it for what is specific to this case, and that they are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._next_forecaster(question).invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The base rate: what the historical record from your research assistant says about typical outcomes and how much they usually move over a similar length of time.
            (b) The time left until the outcome to the question is known.
            (c) The outcome if nothing changed.
            (d) The outcome if the current trend continued.
            (e) The expectations of experts and markets.
            (f) A brief description of an unexpected scenario that results in a low outcome.
            (g) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters start from the base rate and adjust it for what is specific to this case, and that they are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._next_forecaster(question).invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


def make_forecaster(model: str) -> GeneralLlm:
    kwargs: dict = {}
    if model.startswith("anthropic/"):
        # Claude needs room for its thinking plus the written answer.
        kwargs["max_tokens"] = 16000
        if "opus-5-5" in model:
            # Opus 5.5 defaults to medium effort; forecasting deserves high.
            kwargs["output_config"] = {"effort": "high"}
    return GeneralLlm(model=model, timeout=300, allowed_tries=2, **kwargs)


# Provider -> (key variable, cheap model to test the key with).
PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic/claude-haiku-4-5"),
    "openai": ("OPENAI_API_KEY", "openai/gpt-4o-mini"),
}


def working_providers() -> set[str]:
    """
    Providers whose key is set and answers a tiny test call. A mistyped key or
    an empty credit balance on one provider then costs only that provider's
    forecasts instead of failing every question.
    """

    async def ping(model: str) -> str | None:
        try:
            await GeneralLlm(model=model, max_tokens=5, timeout=60).invoke("Reply OK.")
            return None
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:300]}"

    async def ping_all() -> list[str | None]:
        return await asyncio.gather(*(ping(m) for m in candidates.values()))

    candidates = {
        name: model
        for name, (key_var, model) in PROVIDERS.items()
        if _is_real_env(key_var)
    }
    working = set()
    for name, error in zip(candidates, asyncio.run(ping_all())):
        if error:
            logger.error(f"Skipping {name} this run; its API key failed a test call: {error}")
        else:
            working.add(name)
    return working


def uses_working_provider(model: str, providers: set[str]) -> bool:
    provider = model.split("/")[0]
    return provider not in PROVIDERS or provider in providers


def default_forecast_models(providers: set[str]) -> list[str]:
    """One forecasting model per working provider."""
    models = []
    if "anthropic" in providers:
        models.append("anthropic/claude-opus-5-5")
    if "openai" in providers:
        models.append("openai/gpt-6-astra")
    return models


def make_search_llm(providers: set[str]) -> GeneralLlm:
    # The library's OpenAI default researcher (gpt-4o-search-preview) has been
    # retired by OpenAI and fails every question, so choose a live one.
    if "anthropic" in providers:
        return GeneralLlm(
            model="anthropic/claude-sonnet-5",
            max_tokens=4000,
            timeout=180,
            web_search_options={"search_context_size": "medium"},
        )
    return GeneralLlm(model="openai/gpt-5-search-api", timeout=180)


def pick_llms(forecaster: str, providers: set[str]) -> dict:
    """
    Pin current models instead of forecasting-tools' GPT-4o-era defaults,
    using only providers that passed the startup test.
    """
    # The parser reads the final number out of every forecast, so it runs on
    # Anthropic, which holds most of the credit; OpenAI's small balance
    # running dry mid-run then loses only the OpenAI forecasts.
    helper = (
        "anthropic/claude-haiku-4-5"
        if "anthropic" in providers
        else "openai/gpt-4o-mini"
    )
    return {
        "default": make_forecaster(forecaster),
        "summarizer": helper,
        "researcher": make_search_llm(providers),
        "parser": helper,
    }


def asknews_configured() -> bool:
    return _is_real_env("ASKNEWS_API_KEY") or (
        _is_real_env("ASKNEWS_CLIENT_ID") and _is_real_env("ASKNEWS_SECRET")
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forecast without submitting anything to Metaculus; save reports to ./reports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="test_questions mode only: forecast just the first N practice questions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("FORECAST_MODEL") or None,
        help="Forecasting model(s), comma-separated to take turns, e.g. "
        "anthropic/claude-opus-5-5,openai/gpt-6-astra "
        "(default: FORECAST_MODEL env var, else one model per provider key)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = not args.dry_run
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)
    providers = working_providers()
    if not providers:
        raise SystemExit(
            "No AI provider passed the startup test (see errors above). "
            "Check the OPENAI_API_KEY / ANTHROPIC_API_KEY secrets and credit balances."
        )
    forecast_models = (
        [m.strip() for m in args.model.split(",") if m.strip()]
        if args.model
        else default_forecast_models(providers)
    )
    forecast_models = [m for m in forecast_models if uses_working_provider(m, providers)]
    if not forecast_models:
        raise SystemExit(
            f"None of the chosen forecasting models run on a working provider ({', '.join(sorted(providers))})."
        )
    llms = pick_llms(forecast_models[0], providers)
    # One model: the template's 5 forecasts. Several: 2 each, so the median
    # blends them evenly.
    forecasts_per_question = 5 if len(forecast_models) == 1 else 2 * len(forecast_models)
    print(
        f"Forecasting models: {', '.join(forecast_models)} "
        f"({forecasts_per_question} forecasts per question)\n"
        f"Research: web search + base rates{' + AskNews' if asknews_configured() else ''}\n"
    )

    # The locked forecasting-tools (0.2.92) still points its "current" IDs at
    # the Summer 2026 season, so pin the Fall 2026 tournaments here.
    FALL_2026_TOURNAMENT_ID = 33121  # https://www.metaculus.com/tournament/fall-futureeval-2026/
    METACULUS_CUP_FALL_2026_ID = 33108  # https://www.metaculus.com/tournament/metaculus-cup-fall-2026/

    template_bot = SummerTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=forecasts_per_question,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=(
            "reports/" + "+".join(m.split("/")[-1] for m in forecast_models)
            if args.dry_run
            else None
        ),
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=llms,
    )
    if len(forecast_models) > 1:
        template_bot.forecasters = [make_forecaster(m) for m in forecast_models]
    template_bot.base_rate_researcher = make_search_llm(providers)
    template_bot.use_asknews = asknews_configured()

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                FALL_2026_TOURNAMENT_ID, return_exceptions=True
            )
        )
        # MiniBench is off unless INCLUDE_MINIBENCH is set: its prizes are
        # small and it roughly doubles the number of questions to pay for.
        if os.getenv("INCLUDE_MINIBENCH", "").strip().lower() in ("1", "true", "yes"):
            minibench_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    client.CURRENT_MINIBENCH_ID, return_exceptions=True
                )
            )
        else:
            minibench_reports = []
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). AXC_2025_TOURNAMENT_ID = 32564 and
        # AI_2027_TOURNAMENT_ID = "ai-2027" are also valid targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                METACULUS_CUP_FALL_2026_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        if args.limit:
            practice_questions = client.get_all_open_questions_from_tournament(
                "bot-testing-area"
            )[: args.limit]
            forecast_reports = asyncio.run(
                template_bot.forecast_questions(
                    practice_questions, return_exceptions=True
                )
            )
        else:
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    "bot-testing-area", return_exceptions=True
                )
            )

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
